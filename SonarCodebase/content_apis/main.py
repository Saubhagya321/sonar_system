import logging
from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from models.content_models.item_content import ItemContent
from models.content_models.user_content_model import UserContent
import config as AppConfig


###### Logging — single configuration for the whole application

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



###### Schemas

class RecommendationFilters(BaseModel):
    """Optional business-rule filters that the caller can apply on results."""
    chauffeur_active: Optional[bool]  = Field(None, description="Filter by chauffeur active status")
    vehicle_active: Optional[bool]    = Field(None, description="Filter by vehicle active status")
    vehicle_type_id: Optional[int]    = Field(None, description="Filter by vehicle type ID")
    no_of_pax: Optional[int]          = Field(None, description="Minimum seat capacity required")
    red_zone_capable: Optional[int]   = Field(None, description="Filter by red zone capability")
    baby_seat_training: Optional[int] = Field(None, description="Filter by baby seat training")
    client_sensitivity_level: Optional[int] = Field(None, description="Minimum sensitivity level required")
    port_id: Optional[int]            = Field(None, description="Filter to vehicles/chauffeurs from this port only")


class UserRecommendationRequest(BaseModel):
    client_id: int                          = Field(..., gt=0, description="Client ID for recommendations")
    filters: Optional[RecommendationFilters] = None
    candidate_k: Optional[int]             = Field(AppConfig.DEFAULT_CANDIDATE_K, description="Initial candidate window size")
    final_k: Optional[int]                 = Field(AppConfig.DEFAULT_FINAL_K,     description="Final number of results to return")


class ItemRecommendationRequest(BaseModel):
    ref_vehicle: int                         = Field(..., gt=0, description="Vehicle reference ID")
    ref_chauffeur: Optional[int]             = Field(None, gt=0, description="Chauffeur reference ID")
    filters: Optional[RecommendationFilters] = None
    candidate_k: Optional[int]              = Field(AppConfig.DEFAULT_CANDIDATE_K, description="Initial candidate window size")
    final_k: Optional[int]                  = Field(AppConfig.DEFAULT_FINAL_K,     description="Final number of results to return")


class RecommendationResponse(BaseModel):
    success: bool
    message: str
    data: Optional[list]  = None
    count: Optional[int]  = None


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    models_loaded: bool
    endpoints: list[str]



# FilterService 

class FilterService:
    """
    Applies business-rule filters to a recommendations DataFrame.

    Responsibilities:
        - apply()          : apply each filter field to the DataFrame
        - get_filtered_top(): sliding-window expansion to reach final_k results
        - _add_pair_id()   : build the chauffeur_vehicle pair identifier
    """

    # ── Column name constants so string literals don't leak everywhere ──────
    COL_CHAUFFEUR_ACTIVE:    str = "isActive"
    COL_VEHICLE_ACTIVE:      str = "active_status"
    COL_VEHICLE_TYPE:        str = "RefVehicleType"
    COL_SEAT_CAPACITY:       str = "SeatCapacity"
    COL_RED_ZONE:            str = "red_zone_capable"
    COL_BABY_SEAT:           str = "baby_Seat_training"
    COL_CHAUFFEUR_SENS:      str = "ChauffeurRefSensitivityLevel"
    COL_VEHICLE_SENS:        str = "VehicleRefSensitivityLevel"
    COL_REF_CHAUFFEUR:       str = "RefChauffeur"
    COL_REF_VEHICLE:         str = "RefVehicle"

    def apply(self, df: pd.DataFrame, filters: Optional[RecommendationFilters]) -> pd.DataFrame:
        """
        Apply every non-None filter field to the DataFrame.
        Each filter is isolated so one failure never blocks the others.

        Args:
            df      : DataFrame of raw recommendation rows.
            filters : Caller-supplied filter configuration.

        Returns:
            Filtered copy of the DataFrame.
        """
        if filters is None or df.empty:
            return df

        result = df.copy()
        result = self._filter_chauffeur_active(result, filters.chauffeur_active)
        result = self._filter_vehicle_active(result, filters.vehicle_active)
        result = self._filter_vehicle_type(result, filters.vehicle_type_id)
        result = self._filter_seat_capacity(result, filters.no_of_pax)
        result = self._filter_red_zone(result, filters.red_zone_capable)
        result = self._filter_baby_seat(result, filters.baby_seat_training)
        result = self._filter_sensitivity(result, filters.client_sensitivity_level)
        return result

    def get_filtered_top(
        self,
        df: pd.DataFrame,
        filters: Optional[RecommendationFilters],
        candidate_k: int,
        final_k: int,
        max_k: int = AppConfig.MAX_CANDIDATE_WINDOW,
    ) -> pd.DataFrame:
        """
        Return up to `final_k` filtered rows using a sliding-window strategy.

        Strategy:
            1. Sort by the best available score column.
            2. Take the top `candidate_k` rows and apply filters.
            3. If fewer than `final_k` rows survive, double the window and retry.
            4. Stop when we have enough rows, exhaust the data, or hit `max_k`.

        Args:
            df          : Full ranked DataFrame from the model.
            filters     : Business-rule filters to apply.
            candidate_k : Initial window size.
            final_k     : Desired number of output rows.
            max_k       : Hard ceiling on window expansion.

        Returns:
            DataFrame with at most `final_k` rows, reset index.
        """
        # Defensive bounds
        candidate_k = max(1, int(candidate_k or AppConfig.DEFAULT_CANDIDATE_K))
        final_k     = max(1, int(final_k     or AppConfig.DEFAULT_FINAL_K))
        max_k       = max(candidate_k, int(max_k))

        if df.empty:
            return df.copy()

        df_sorted = self._sort_by_score(df)
        total     = len(df_sorted)

        # Initial window
        k        = min(candidate_k, total)
        filtered = self.apply(df_sorted.head(k), filters)

        if len(filtered) >= final_k:
            return filtered.head(final_k).reset_index(drop=True)

        # Expand window until we have enough results or hit limits
        while k < min(max_k, total):
            k        = min(k * AppConfig.WINDOW_EXPANSION_FACTOR, total, max_k)
            filtered = self.apply(df_sorted.head(k), filters)
            logger.debug("Window expanded to %d → %d rows after filter", k, len(filtered))

            if len(filtered) >= final_k or k >= total or k >= max_k:
                break

        return filtered.head(final_k).reset_index(drop=True)

    def add_pair_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a `pair_id` column in the format '<chauffeur_id>_<vehicle_id>'.
        Operates in-place on a copy. Skips silently if columns are missing.

        Args:
            df: DataFrame that may contain RefChauffeur and RefVehicle columns.

        Returns:
            DataFrame with pair_id column added (or unchanged if not possible).
        """
        has_chauffeur = self.COL_REF_CHAUFFEUR in df.columns
        has_vehicle   = self.COL_REF_VEHICLE   in df.columns

        if not (has_chauffeur and has_vehicle):
            logger.warning("Cannot create pair_id — missing %s or %s",
                           self.COL_REF_CHAUFFEUR, self.COL_REF_VEHICLE)
            return df

        try:
            df = df.copy()
            df["pair_id"] = (
                df[self.COL_REF_CHAUFFEUR].astype(int).astype(str)
                + "_"
                + df[self.COL_REF_VEHICLE].astype(int).astype(str)
            )
            logger.debug("pair_id sample: %s", df["pair_id"].head(3).tolist())
        except Exception as exc:
            logger.warning("Could not create pair_id: %s", exc)

        return df

    # ── Private filter helpers

    def _filter_chauffeur_active(self, df: pd.DataFrame, value: Optional[bool]) -> pd.DataFrame:
        if value is not None and self.COL_CHAUFFEUR_ACTIVE in df.columns:
            return df[df[self.COL_CHAUFFEUR_ACTIVE] == int(value)]
        return df

    def _filter_vehicle_active(self, df: pd.DataFrame, value: Optional[bool]) -> pd.DataFrame:
        if value is not None and self.COL_VEHICLE_ACTIVE in df.columns:
            return df[df[self.COL_VEHICLE_ACTIVE] == int(value)]
        return df

    def _filter_vehicle_type(self, df: pd.DataFrame, value: Optional[int]) -> pd.DataFrame:
        if value is not None and self.COL_VEHICLE_TYPE in df.columns:
            return df[df[self.COL_VEHICLE_TYPE] == value]
        return df

    def _filter_seat_capacity(self, df: pd.DataFrame, value: Optional[int]) -> pd.DataFrame:
        if value is not None and self.COL_SEAT_CAPACITY in df.columns:
            df = df.copy()
            df[self.COL_SEAT_CAPACITY] = pd.to_numeric(df[self.COL_SEAT_CAPACITY], errors="coerce").fillna(0)
            return df[df[self.COL_SEAT_CAPACITY] >= int(value)]
        return df

    def _filter_red_zone(self, df: pd.DataFrame, value: Optional[int]) -> pd.DataFrame:
        if value is not None and self.COL_RED_ZONE in df.columns:
            return df[df[self.COL_RED_ZONE] == int(value)]
        return df

    def _filter_baby_seat(self, df: pd.DataFrame, value: Optional[int]) -> pd.DataFrame:
        if value is not None and self.COL_BABY_SEAT in df.columns:
            return df[df[self.COL_BABY_SEAT] == int(value)]
        return df

    def _filter_sensitivity(self, df: pd.DataFrame, level: Optional[int]) -> pd.DataFrame:
        if level is None:
            return df
        result = df
        if self.COL_CHAUFFEUR_SENS in result.columns:
            ch = pd.to_numeric(result[self.COL_CHAUFFEUR_SENS], errors="coerce").fillna(0)
            result = result[ch >= int(level)]
        if self.COL_VEHICLE_SENS in result.columns:
            ve = pd.to_numeric(result[self.COL_VEHICLE_SENS], errors="coerce").fillna(0)
            result = result[ve >= int(level)]
        return result

    @staticmethod
    def _sort_by_score(df: pd.DataFrame) -> pd.DataFrame:
        """Sort by the first available score column."""
        for col in AppConfig.SCORE_COLUMNS:
            if col in df.columns:
                return df.sort_values(col, ascending=False)
        return df.copy()



# ModelService 

class ModelService:
    """
    Manages the lifecycle of UserContent and ItemContent models.

    Responsibilities:
        - initialize()         : load both models once at startup
        - get_user_recommendations() : delegate to UserContent.run_model()
        - get_item_recommendations() : delegate to ItemContent.run_model()
        - is_ready property    : lets the health endpoint report truthfully
    """

    def __init__(self) -> None:
        self._user_model: Optional[UserContent] = None
        self._item_model: Optional[ItemContent] = None

    def initialize(self) -> None:
        """
        Instantiate both ML models.
        Called once during app lifespan startup — never on a hot path.
        """
        logger.info("Initializing recommendation models…")
        self._user_model = UserContent()
        self._item_model = ItemContent()
        logger.info("Both models loaded successfully.")

    @property
    def is_ready(self) -> bool:
        """True only when both models have been initialised."""
        return self._user_model is not None and self._item_model is not None

    def get_user_recommendations(self, client_id: int, port_id: Optional[int]) -> pd.DataFrame:
        """
        Run the user-content model for the given client.

        Args:
            client_id : The client whose history drives the recommendation.
            port_id   : Optional port constraint passed straight to the model.

        Returns:
            DataFrame of ranked recommendations, or empty DataFrame on no results.

        Raises:
            RuntimeError: if the model has not been initialized.
        """
        self._assert_ready("user")
        logger.info("User model — client_id=%d, port_id=%s", client_id, port_id)
        result = self._user_model.run_model(client_id=client_id, port_id=port_id)
        return self._validate_model_output(result)

    def get_item_recommendations(
        self,
        ref_vehicle: int,
        ref_chauffeur: Optional[int],
        port_id: Optional[int],
    ) -> pd.DataFrame:
        """
        Run the item-content model for the given vehicle / chauffeur pair.

        Args:
            ref_vehicle   : Vehicle reference ID (required).
            ref_chauffeur : Chauffeur reference ID (optional).
            port_id       : Optional port constraint passed straight to the model.

        Returns:
            DataFrame of ranked recommendations, or empty DataFrame on no results.

        Raises:
            RuntimeError: if the model has not been initialized.
        """
        self._assert_ready("item")
        logger.info(
            "Item model — ref_vehicle=%d, ref_chauffeur=%s, port_id=%s",
            ref_vehicle, ref_chauffeur, port_id,
        )
        result = self._item_model.run_model(
            ref_vehicle=ref_vehicle,
            ref_chauffeur=ref_chauffeur,
            port_id=port_id,
        )
        return self._validate_model_output(result)

    def run_batch_user_recommendations(
        self, client_ids: list[int]
    ) -> dict[int, list | dict]:
        """
        Run the user model for every client in the list.
        Errors per client are caught individually so one bad ID
        never aborts the whole batch.

        Args:
            client_ids: List of client IDs to process.

        Returns:
            Dict mapping each client_id to its recommendations list
            (or an error dict if that client failed).
        """
        self._assert_ready("user")
        results: dict[int, list | dict] = {}

        for client_id in client_ids:
            try:
                df = self._user_model.run_model(client_id=client_id)
                df = self._validate_model_output(df)
                if df.empty:
                    results[client_id] = []
                    continue
                trimmed = (
                    df.sort_values("similarity", ascending=False)
                    .head(AppConfig.DEFAULT_BATCH_FINAL_K)
                    .reset_index(drop=True)
                )
                results[client_id] = trimmed.to_dict(orient="records")
            except Exception as exc:
                logger.warning("Batch error for client_id=%d: %s", client_id, exc)
                results[client_id] = {"error": str(exc)}

        return results

    # ── Private helpers

    def _assert_ready(self, model_name: str) -> None:
        if not self.is_ready:
            raise RuntimeError(f"{model_name.title()} model is not initialized.")

    @staticmethod
    def _validate_model_output(result) -> pd.DataFrame:
        """
        Normalise model output:
            - None           → empty DataFrame
            - non-DataFrame  → empty DataFrame
            - Message column → empty DataFrame  (model signalled no results)
            - valid          → return as-is
        """
        if result is None or not isinstance(result, pd.DataFrame):
            return pd.DataFrame()
        if result.empty or "Message" in result.columns:
            return pd.DataFrame()
        return result


# RecommendationRouter 

class RecommendationRouter:
    """
    Defines and handles all /recommendations/* routes.

    Receives ModelService and FilterService via constructor injection
    so each dependency can be swapped or mocked independently in tests.
    """

    def __init__(self, model_service: ModelService, filter_service: FilterService) -> None:
        self._models  = model_service
        self._filters = filter_service

    # ── Route handlers ───────────────────────────────────────────────────────

    def user_recommendations(
        self, request: UserRecommendationRequest
    ) -> RecommendationResponse:
        """
        GET recommendations for a single client based on interaction history.
        """
        self._check_ready()

        client_id   = request.client_id
        candidate_k = int(request.candidate_k or AppConfig.DEFAULT_CANDIDATE_K)
        final_k     = int(request.final_k     or AppConfig.DEFAULT_FINAL_K)
        port_id     = request.filters.port_id if request.filters else None

        raw_df = self._models.get_user_recommendations(client_id=client_id, port_id=port_id)
        if raw_df.empty:
            return self._empty_response(f"No recommendations found for client {client_id}")
       
        final_df = self._filters.get_filtered_top(raw_df, request.filters, candidate_k, final_k)
        
        if final_df.empty:
            return self._empty_response(
                f"No recommendations for client {client_id} after applying filters"
            )

        final_df = self._filters.add_pair_id(final_df)
        # print(f"see final response should be {final_df}")
        return self._success_response(final_df)

    def item_recommendations(
        self, request: ItemRecommendationRequest
    ) -> RecommendationResponse:
        """
        GET recommendations for a vehicle / chauffeur pair.
        """
        self._check_ready()

        candidate_k = int(request.candidate_k or AppConfig.DEFAULT_CANDIDATE_K)
        final_k     = int(request.final_k     or AppConfig.DEFAULT_FINAL_K)
        port_id     = request.filters.port_id if request.filters else None

        raw_df = self._models.get_item_recommendations(
            ref_vehicle=request.ref_vehicle,
            ref_chauffeur=request.ref_chauffeur,
            port_id=port_id,
        )
        if raw_df.empty:
            return self._empty_response("No recommendations found for this vehicle-chauffeur pair")

        final_df = self._filters.get_filtered_top(raw_df, request.filters, candidate_k, final_k)
        if final_df.empty:
            return self._empty_response("No recommendations found after applying filters")

        final_df = self._filters.add_pair_id(final_df)
        return self._success_response(final_df)

    def batch_user_recommendations(
        self, client_ids: list[int]
    ) -> dict:
        """
        GET recommendations for multiple clients in one call.
        """
        self._check_ready()
        results = self._models.run_batch_user_recommendations(client_ids)
        return {
            "success": True,
            "message": f"Processed {len(client_ids)} clients",
            "data": results,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _check_ready(self) -> None:
        """Raise 503 if models are not yet loaded."""
        if not self._models.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Recommendation models are not yet initialized.",
            )

    @staticmethod
    def _success_response(df: pd.DataFrame) -> RecommendationResponse:
        records = df.to_dict(orient="records")
        logger.info("Returning %d recommendations.", len(records))
        return RecommendationResponse(
            success=True,
            message=f"Successfully retrieved {len(records)} recommendations",
            data=records,
            count=len(records),
        )

    @staticmethod
    def _empty_response(message: str) -> RecommendationResponse:
        logger.info("Empty result: %s", message)
        return RecommendationResponse(success=False, message=message, data=[], count=0)




# Instantiate services once (singletons for the process lifetime)
_model_service  = ModelService()
_filter_service = FilterService()
_router         = RecommendationRouter(_model_service, _filter_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: initialize models on startup, log on shutdown."""
    try:
        _model_service.initialize()
    except Exception as exc:
        logger.critical("Failed to initialize models: %s", exc, exc_info=True)
        raise
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title=AppConfig.TITLE,
    description=AppConfig.DESCRIPTION,
    version=AppConfig.VERSION,
    lifespan=lifespan,
)


# ── Health routes ────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], response_model=HealthResponse)
async def root() -> HealthResponse:
    """Basic liveness probe."""
    return HealthResponse(
        status="running",
        message=AppConfig.TITLE,
        version=AppConfig.VERSION,
        models_loaded=_model_service.is_ready,
        endpoints=[
            "/recommendations/user",
            "/recommendations/item",
            "/recommendations/user/batch",
        ],
    )


@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Detailed readiness probe — returns 503 if models are not loaded."""
    if not _model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not yet initialized.",
        )
    return HealthResponse(
        status="healthy",
        message="All systems operational.",
        version=AppConfig.VERSION,
        models_loaded=True,
        endpoints=[
            "/recommendations/user",
            "/recommendations/item",
            "/recommendations/user/batch",
        ],
    )


# ── Recommendation routes
@app.post(
    "/recommendations/user",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="User-based recommendations",
)
async def user_recommendations(request: UserRecommendationRequest) -> RecommendationResponse:
    """Return ranked vehicle-chauffeur pairs for a given client."""
    try:
        return _router.user_recommendations(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /recommendations/user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {exc}",
        )


@app.post(
    "/recommendations/item",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Item-based recommendations",
)
async def item_recommendations(request: ItemRecommendationRequest) -> RecommendationResponse:
    """Return ranked vehicle-chauffeur pairs similar to a given pair."""
    try:
        return  _router.item_recommendations(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /recommendations/item")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {exc}",
        )


@app.post(
    "/recommendations/user/batch",
    tags=["Recommendations"],
    summary="Batch user recommendations",
)
async def batch_user_recommendations(client_ids: list[int]) -> dict:
    """Return recommendations for multiple clients in a single request."""
    try:
        return  _router.batch_user_recommendations(client_ids)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /recommendations/user/batch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing error: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=AppConfig.HOST, port=AppConfig.PORT)
