import logging
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Any
from datetime import datetime
from config import DB_CONNECTION_STRING

logger = logging.getLogger(__name__)


class DatabaseOperations:
    """Handle all database operations including stored procedures"""
    
    def __init__(self, connection_string: str = DB_CONNECTION_STRING):
        self.connection_string = connection_string
        # Connection pooling - create engine once, reuse connections
        self.engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True  # Verify connections before use
        )
        logger.info("DatabaseOperations initialized with connection pooling")
    
    def execute_allocation_procedure(self, batch_id: str) -> Dict[str, Any]:
        """
        Execute the allocation stored procedure for a given batch_id
        
        Args:
            batch_id: The batch ID to process allocations for
            
        Returns:
            Dict with execution results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"\n{'='*70}")
            logger.info("EXECUTING ALLOCATION PROCEDURE")
            logger.info(f"Batch ID: {batch_id}")
            logger.info(f"{'='*70}")
            
            # Use pooled connection instead of creating new engine
            with self.engine.connect() as conn:
                # Execute stored procedure
                query = text("EXEC dbo.allocate_by_recommendation :batch_id")
                result = conn.execute(query, {'batch_id': batch_id})
                conn.commit()
                
                # Try to fetch results if the SP returns any
                try:
                    rows = result.fetchall()
                    result_data = [dict(row._mapping) for row in rows] if rows else []
                except Exception:
                    # Some SPs don't return results
                    result_data = []
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.info("\n  Allocation procedure completed successfully")
                logger.info(f"Execution time: {execution_time:.2f}s")
                if result_data:
                    logger.info(f"Rows affected/returned: {len(result_data)}")
                logger.info(f"{'='*70}\n")
                
                return {
                    "success": True,
                    "batch_id": batch_id,
                    "execution_time_seconds": execution_time,
                    "rows_affected": len(result_data),
                    "result_data": result_data,
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            logger.error("\n  ALLOCATION PROCEDURE FAILED")
            logger.error(f"Batch ID: {batch_id}")
            logger.error(f"Error: {error_msg}")
            logger.error(f"Execution time: {execution_time:.2f}s")
            logger.error(f"{'='*70}\n", exc_info=True)
            
            return {
                "success": False,
                "batch_id": batch_id,
                "error": error_msg,
                "execution_time_seconds": execution_time,
                "timestamp": datetime.now().isoformat()
            }
        
        # REMOVED: finally block with engine.dispose() - pooling handles this
    
    def execute_populate_availability(self, batch_id: str = None, booking_id: int = None, top_n: int = 10) -> Dict[str, Any]:
        """
        Execute the PopulateAvailabilityAllocation stored procedure
        
        Args:
            batch_id: Optional batch ID to process
            booking_id: Optional specific booking ID to process
            top_n: Number of top recommendations per booking (default 10)
            
        Returns:
            Dict with execution results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"\n{'='*70}")
            logger.info("  POPULATING AVAILABILITY ALLOCATION")
            logger.info(f"   Batch ID: {batch_id}")
            logger.info(f"   Booking ID: {booking_id}")
            logger.info(f"   Top N: {top_n}")
            logger.info(f"{'='*70}")
            
            # Use pooled connection instead of creating new engine
            with self.engine.connect() as conn:
                # Execute stored procedure
                query = text("""
                    EXEC dbo.PopulateAvailabilityAllocation 
                        @BatchId = :batch_id,
                        @BookingID = :booking_id,
                        @TopN = :top_n
                """)
                result = conn.execute(query, {
                    'batch_id': batch_id,
                    'booking_id': booking_id,
                    'top_n': top_n
                })
                conn.commit()
                
                # Fetch results
                try:
                    rows = result.fetchall()
                    result_data = [dict(row._mapping) for row in rows] if rows else []
                except Exception:
                    result_data = []
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.info("\n  Populate availability completed")
                logger.info(f"   Execution time: {execution_time:.2f}s")
                if result_data:
                    logger.info(f"   Result: {result_data}")
                logger.info(f"{'='*70}\n")
                
                return {
                    "success": True,
                    "batch_id": batch_id,
                    "booking_id": booking_id,
                    "top_n": top_n,
                    "execution_time_seconds": execution_time,
                    "result_data": result_data,
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            logger.error("\n  POPULATE AVAILABILITY FAILED")
            logger.error(f"   Error: {error_msg}")
            logger.error(f"{'='*70}\n", exc_info=True)
            
            return {
                "success": False,
                "batch_id": batch_id,
                "booking_id": booking_id,
                "error": error_msg,
                "execution_time_seconds": execution_time,
                "timestamp": datetime.now().isoformat()
            }
        
        # REMOVED: finally block with engine.dispose() - pooling handles this
    
    def dispose(self):
        """Dispose of the connection pool (call on shutdown)"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection pool disposed")


# Create a singleton instance
db_ops = DatabaseOperations()