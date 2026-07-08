export interface Booking {
  id: string;
  port: string;
  bookingNumber: string;
  pickupTime: string;
  pickupDate: string;
  pickupAddress: string;
  dropoffAddress: string;
  clientAccountNumber: string;
  requestedVehicleType: string;
  assignedVehicle: string;
  assignedChauffeur: string;
  status: 'Allocated' | 'Pending' | 'Rejected';
   top_recommendation?: {
    vehicle_id: string;
    chauffeur_id: string;
    pair_id: string;
    cf_score?: number;
    final_score?: number;
    rank?: number;
  };
}

export interface GroupedBooking {
  groupKey: string;
  groupValue: string;
  bookings: Booking[];
}
