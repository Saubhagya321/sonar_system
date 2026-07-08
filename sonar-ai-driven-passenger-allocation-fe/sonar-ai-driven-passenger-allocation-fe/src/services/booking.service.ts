import axios from 'axios';
import { Booking } from '@/types/booking';

const API_URL = import.meta.env.VITE_API_URL;

export const fetchPendingBookings = async (): Promise<Booking[]> => {
  const response = await axios.get(
    `${API_URL}/feedback/pending-bookings`,
    {
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    }
  );

  return response.data.bookings.map((item: any): Booking => ({
    id: String(item.booking_id),
    port: item.port,
    bookingNumber: item.booking_number,
    pickupTime: item.pickup_time,
    pickupDate: item.pickup_date,
    pickupAddress: item.pickup_address,
    dropoffAddress: item.dropoff_address,
    clientAccountNumber: item.client_account_number,
    requestedVehicleType: item.requested_vehicle_type,
    assignedVehicle: item.assigned_vehicle,
    assignedChauffeur: item.assigned_chauffeur,
    status: 'AI Allocated',
    top_recommendation: item.top_recommendation,
  }));
};
