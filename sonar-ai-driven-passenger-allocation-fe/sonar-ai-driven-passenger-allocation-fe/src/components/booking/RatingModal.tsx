import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import axios from 'axios';
import { toast } from 'react-toastify';

interface RatingModalProps {
  isOpen: boolean;
  booking: Booking | null;
  onClose: () => void;
  onSuccess?: () => void; // optional callback to refresh bookings
}


const API_URL = import.meta.env.VITE_API_URL;

export const RatingModal: React.FC<RatingModalProps> = ({
  isOpen,
  booking,
  onClose,
  onSuccess,
}) => {
  const [selectedRating, setSelectedRating] = React.useState<number>(0);
  const [loading, setLoading] = React.useState(false);

  const handleSubmit = async () => {
  // if (!selectedRating || !booking?.top_recommendation) return;

  const topRec = booking.top_recommendation;
  const selectedPairId = topRec.pair_id;

  setLoading(true);
  try {
    await axios.post(`${API_URL}/feedback/select`, {
      request_id: booking.id,
      selected_pair_id: selectedPairId,
      overall_quality: selectedRating,
    });
toast.success('Rating submitted successfully!');
    setSelectedRating(0);
    onClose();

    
    if (onSuccess) onSuccess();

  } catch (err) {
    console.error(err);
    toast.error('Failed to submit rating'); 
  } finally {
    setLoading(false);
  }
};



  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent
  className="
    w-[500px]
    h-[328px]
    p-0
    bg-white
    rounded-[12px]
    shadow-[0px_2px_2px_rgba(27,36,44,0.12)]
    [&>button]:hidden
  "
>

        {/* Header */}
        <DialogHeader className="px-[20px] pt-[18px] pb-[10px]">
          <div className="flex items-center justify-between">
            <DialogTitle className="text-[20px] font-semibold text-[#1E273B]">
              Rate Your Experience
            </DialogTitle>

            <button
              onClick={onClose}
              className="w-5 h-5 flex items-center justify-center opacity-50"
            >
              ✕
            </button>
          </div>
        </DialogHeader>

        {/* Divider */}
        <div className="h-px bg-[#E8ECEF]" />

        {/* Body */}
        <div className="px-[50px] py-[24px] flex flex-col items-center gap-[20px]">
          {/* Text */}
          <div className="flex flex-col items-center gap-[10px]">
            <p className="text-[18px] font-medium text-[#434B50] text-center">
              How would you rate our recommendations?
            </p>
            <p className="text-[16px] text-[#849098] text-center">
              Your feedback helps us serve you better
            </p>
          </div>

          {/* Rating buttons */}
          <div className="flex gap-[15px]">
            {[1, 2, 3, 4, 5].map((value) => {
              const isActive = selectedRating === value;

              return (
                <button
                  key={value}
                  onClick={() => setSelectedRating(value)}
                  className={`
                    w-[48px] h-[48px]
                    rounded-[9px]
                    flex items-center justify-center
                    text-[19.5px] font-medium
                    transition
                    ${
                      isActive
                        ? 'bg-[#0033A0] text-white'
                        : 'bg-[#F0F3F6] text-[#434B50]'
                    }
                  `}
                >
                  {value}
                </button>
              );
            })}
          </div>

          {/* Submit button */}
          <Button
            onClick={handleSubmit}
            disabled={!selectedRating || loading}
            className={`
              w-full h-[48px]
              rounded-[8px]
              text-[16px] font-semibold
              ${
                selectedRating
                  ? 'bg-[#0033A0] text-white hover:bg-[#002780]'
                  : 'bg-[#F0F3F6] text-[#B5BCC1]'
              }
            `}
          >
            {loading ? 'Submitting...' : 'Submit Rating'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
