import React from 'react';
import { Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Calendar as CalendarComponent } from '@/components/ui/calendar';
import { format } from 'date-fns';

interface PageTitleSectionProps {
  totalResults: number;
  selectedDate: Date | undefined;
  onDateChange: (date: Date | undefined) => void;
  handleReset: () => void;
}

export const PageTitleSection: React.FC<PageTitleSectionProps> = ({
  totalResults,
  selectedDate,
  onDateChange,
  handleReset
}) => {
  return (
    <div className="flex justify-between items-center mb-3">
      <h1 className="text-[#1A1A1A] text-[18px] leading-[24px] font-semibold">
        Booking Management{' '}
        <span className="">
          ({totalResults} Results)
        </span>
      </h1>

      <div className="flex items-center">
        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              className="h-9 px-3 text-sm border-[#E0E0E0] text-[#666666]"
            >
              <Calendar className="mr-2 h-4 w-4" />
              {selectedDate
                ? format(selectedDate, 'dd MMMM yyyy')
                : 'Select Date'}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0" align="end">
            <CalendarComponent
              mode="single"
              selected={selectedDate}
              onSelect={onDateChange}
              initialFocus
              disabled={(date) => {
                const today = new Date();
                today.setHours(0, 0, 0, 0);

                const tomorrow = new Date(today);
                tomorrow.setDate(today.getDate() + 1);

                const selected = new Date(date);
                selected.setHours(0, 0, 0, 0);

                return selected < today || selected > tomorrow;
              }}
            />

          </PopoverContent>
        </Popover>
        <Button
          variant="outline"
          className="h-9 px-3 ml-2 text-sm border-[#E0E0E0] text-[#666666]"
          onClick={() => handleReset()}
        >
          Reset
        </Button>
      </div>
    </div>
  );
};

