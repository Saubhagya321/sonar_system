import React, { useEffect, useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { X } from 'lucide-react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { cn } from '@/lib/utils';
import { Booking } from '@/types/booking';

interface RejectModalProps {
  isOpen: boolean;
  booking: Booking | null;
  onClose: () => void;
  onSuccess?: () => void;
}

const API_URL = import.meta.env.VITE_API_URL;

const REJECTION_CATEGORIES = [
  { label: 'Client Preference', code: 'CLIENT_PREFERENCE' },
  { label: 'Vehicle Mismatch', code: 'VEHICLE_MISMATCH' },
  { label: 'Poor Match', code: 'POOR_MATCH' },
  { label: 'Past Negative Experience', code: 'PAST_EXPERIENCE' },
  { label: 'Comfort Preference', code: 'COMFORT' },
  { label: 'Reliability Concern', code: 'RELIABILITY' },
  { label: 'Service Expectation', code: 'SERVICE_EXPECTATION' },
  { label: 'Chauffeur Unavailable', code: 'CHAUFFEUR_UNAVAILABLE' },
  { label: 'Vehicle Unavailable', code: 'VEHICLE_UNAVAILABLE' },
  { label: 'External Factor', code: 'EXTERNAL_FACTOR' },
];

export const RejectModal: React.FC<RejectModalProps> = ({
  isOpen,
  booking,
  onClose,
  onSuccess,
}) => {
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isOpen) {
      setSelectedCategories([]);
    }
  }, [isOpen]);

  const toggleCategory = (code: string) => {
    setSelectedCategories((prev) =>
      prev.includes(code)
        ? prev.filter((c) => c !== code)
        : [...prev, code]
    );
  };

  const handleSubmit = async () => {
    // if (!booking?.top_recommendation || selectedCategories.length === 0) return;

    setLoading(true);
    try {
      await axios.post(`${API_URL}/feedback/reject`, {
        request_id: booking.id,
        pair_id: booking.top_recommendation.pair_id,
        categories: selectedCategories,
        reason_text: '',
      });

      toast.success('Booking rejected successfully');
      onClose();
      onSuccess?.();
    } catch (err) {
      console.error(err);
      toast.error('Failed to reject booking');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent
        className="
          w-[560px]
          p-0
          bg-white
          rounded-[12px]
          [&>button]:hidden
        "
      >
        {/* Header */}
        <DialogHeader className="px-6 pt-5 pb-3">
          <div className="flex items-center justify-between">
            <DialogTitle className="text-[18px] font-semibold text-[#1E273B]">
              Booking Rejected
            </DialogTitle>
            <button onClick={onClose} className="text-[#6B7280] hover:text-black">
              <X size={18} />
            </button>
          </div>
          <p className="text-sm text-[#6B7280] mt-1">
            Please select one or more reasons for rejecting this booking.
          </p>
        </DialogHeader>

        <div className="h-px bg-[#E8ECEF]" />

        {/* Body */}
        <div className="px-6 py-4 flex flex-wrap gap-3">
          {REJECTION_CATEGORIES.map(({ label, code }) => {
            const selected = selectedCategories.includes(code);
            return (
              <button
                key={code}
                onClick={() => toggleCategory(code)}
                className={cn(
                  'flex items-center justify-between gap-3 px-4 py-2 rounded-lg border text-sm transition-all min-w-[170px]',
                  selected
                    ? 'border-[#2D5BFF] bg-[#F5F8FF] text-[#2D5BFF]'
                    : 'border-[#E8ECEF] text-[#1E273B] hover:border-[#2D5BFF]'
                )}
              >
                <span>{label}</span>

                {/* Circle indicator (right side like Figma) */}
                <span
                  className={cn(
                    'w-4 h-4 rounded-full border flex items-center justify-center',
                    selected
                      ? 'border-[#2D5BFF]'
                      : 'border-[#CBD5E1]'
                  )}
                >
                  {selected && (
                    <span className="w-2 h-2 rounded-full bg-[#2D5BFF]" />
                  )}
                </span>
              </button>
            );
          })}
        </div>

        <div className="h-px bg-[#E8ECEF]" />

        {/* Footer */}
        <div className="px-6 py-4 flex justify-end gap-3">
          <Button
            variant="ghost"
            onClick={onClose}
            className="text-[#1E273B]"
          >
            Cancel
          </Button>

          <Button
            onClick={handleSubmit}
            disabled={loading || selectedCategories.length === 0}
            className={cn(
              'px-6 bg-[#2D5BFF] text-white',
              loading && 'opacity-60 cursor-not-allowed'
            )}
          >
            {loading ? 'Submitting...' : 'Submit'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
