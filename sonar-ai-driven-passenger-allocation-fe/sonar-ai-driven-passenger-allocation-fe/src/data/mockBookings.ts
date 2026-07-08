import { Booking } from '@/types/booking';

export const mockBookings: Booking[] = Array.from({ length: 25 }, (_, i) => {
  const ports = ['Sydney', 'Auckland', 'Melbourne', 'Adelaide', 'Perth'];
  const pickupAddresses = [
    'Ash Dr, San Jose',
    'Main St, New York',
    'Oak Ave, Los Angeles',
    'Pine Rd, Chicago',
    'Market St, San Francisco',
  ];
  const dropoffAddresses = [
    'Preston Rd, Inglewood',
    'Sunset Blvd, Hollywood',
    'Michigan Ave, Downtown',
    'Broadway St, Midtown',
    'Wall St, Manhattan',
  ];
  const vehicleTypes = ['Premium SUV', 'Executive Sedan', 'Luxury Van'];
  const vehicles = [
    'BSQ - Premium SUV',
    'BSQ - Executive Sedan',
    'BSQ - Luxury Van',
  ];
  const chauffeurs = [
    'NDIN - NECMETTIN DINCER',
    'JOHN - JOHN SMITH',
    'MARY - MARY JOHNSON',
    'ALEX - ALEX BROWN',
  ];

  const date = new Date();
  date.setDate(date.getDate() - (i % 5));

  return {
    id: `booking-${i + 1}`,
    port: ports[i % ports.length],
    bookingNumber: `BK-${1000 + i}`,
    pickupTime: `${String(8 + (i % 6)).padStart(2, "0")}:${i % 2 === 0 ? "00" : "30"}`,
    pickupDate: date.toISOString().split('T')[0],
    pickupAddress: pickupAddresses[i % pickupAddresses.length],
    dropoffAddress: dropoffAddresses[i % dropoffAddresses.length],
    clientAccountNumber: `AC-${2000 + i}`,
    requestedVehicleType: vehicleTypes[i % vehicleTypes.length],
    assignedVehicle: vehicles[i % vehicles.length],
    assignedChauffeur: chauffeurs[i % chauffeurs.length],
    status: 'AI Allocated',
  };
});
