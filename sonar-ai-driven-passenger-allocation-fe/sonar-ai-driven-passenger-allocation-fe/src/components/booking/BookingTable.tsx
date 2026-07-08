import React from 'react';
import { ArrowUpDown, GripVertical, ChevronRight, ChevronDown, Calendar, Clock } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Booking, GroupedBooking } from '@/types/booking';
import { Popover, PopoverTrigger, PopoverContent } from '@radix-ui/react-popover';
import { format } from 'date-fns';
import { Calendar as CalendarComponent } from '@/components/ui/calendar';

interface BookingTableProps {
  bookings: Booking[] | GroupedBooking[];
  groupedColumns: string[];
  sortColumn: string | null;
  sortDirection: 'asc' | 'desc';
  onSort: (column: string) => void;
  onDragStart: (e: React.DragEvent, columnId: string) => void;
  onAccept: (booking: Booking) => void;
  onReject: (booking: Booking) => void;
  columnFilters: Record<string, string>;
  setColumnFilters: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  selectedGridDate: Date | undefined;
  setSelectedGridDate: React.Dispatch<React.SetStateAction<Date | undefined>>;
  selectedTime: string;
  setSelectedTime: React.Dispatch<React.SetStateAction<string>>;
}

interface Column {
  id: keyof Booking | 'actions';
  label: string;
  className?: string;
}
type DragPayload =
  | { type: 'reorder'; column: Column }
  | { type: 'group'; columnId: string };

const columns: Column[] = [
  { id: 'port', label: 'Port' },
  { id: 'bookingNumber', label: 'Booking Number' },
  { id: 'pickupTime', label: 'Pickup Time' },
  { id: 'pickupDate', label: 'Pickup Date' },
  { id: 'pickupAddress', label: 'Pickup Address' },
  { id: 'dropoffAddress', label: 'Dropoff Address' },
  { id: 'clientAccountNumber', label: 'Client Account Number' },
  { id: 'requestedVehicleType', label: 'Requested Vehicle Type' },
  { id: 'assignedVehicle', label: 'Assigned Vehicle' },
  { id: 'assignedChauffeur', label: 'Assigned Chauffeur' },
  { id: 'status', label: 'Status' },
  { id: 'actions', label: 'ACTIONS', className: 'text-right' },
];

export const BookingTable: React.FC<BookingTableProps> = ({
  bookings,
  groupedColumns,
  sortColumn,
  sortDirection,
  onSort,
  onDragStart,
  onAccept,
  onReject,
  columnFilters,
  setColumnFilters,
  selectedGridDate,
  setSelectedGridDate,
  selectedTime,
  setSelectedTime
}) => {
  const [expandedGroups, setExpandedGroups] = React.useState<Set<string>>(new Set());
  const [orderedColumns, setOrderedColumns] = React.useState<Column[]>(columns);
  const dragColumn = React.useRef<Column | null>(null);

  const visibleColumns = React.useMemo(() => {
    return orderedColumns.filter(
      (col) =>
        col.id === 'actions' || !groupedColumns.includes(col.id as string)
    );
  }, [orderedColumns, groupedColumns]);





  /**
   * Function to handle dropping a column header for reordering/suffle columns
   * @param targetColumn Target Column
   * @returns 
   */
  const handleDrop = (targetColumn: Column) => {

    if (!dragColumn.current) return;
    if (dragColumn.current.id === targetColumn.id) return;

    const updatedColumns = [...orderedColumns];
    const fromIndex = updatedColumns.findIndex(
      c => c.id === dragColumn.current!.id
    );
    const toIndex = updatedColumns.findIndex(
      c => c.id === targetColumn.id
    );

    updatedColumns.splice(fromIndex, 1);
    updatedColumns.splice(toIndex, 0, dragColumn.current);

    setOrderedColumns(updatedColumns);
    dragColumn.current = null;
  };



  const toggleGroup = (groupKey: string) => {
    const newExpanded = new Set(expandedGroups);
    if (newExpanded.has(groupKey)) {
      newExpanded.delete(groupKey);
    } else {
      newExpanded.add(groupKey);
    }
    setExpandedGroups(newExpanded);
  };

  const renderCell = (booking: Booking, column: Column) => {
    if (column.id === 'actions') {
      return (
        <div className="flex gap-2 justify-start">
          <Button
            onClick={() => onAccept(booking)}
            className="flex justify-center items-center px-[16px] py-[10px] w-[79px] h-[32px] bg-[#2C45BE] rounded-[8px] font-instrument-sans font-medium text-[14px] text-white"
          >
            Accept
          </Button>

          <Button
            onClick={() => onReject(booking)}
            variant="outline"
            className="flex justify-center items-center px-[16px] py-[10px] w-[74px] h-[32px] bg-[#F0F3F6] rounded-[8px] font-instrument-sans font-medium text-[14px] text-[#2F3438] border-none"
          >
            Reject
          </Button>

        </div>
      );
    }

    if (column.id === 'status') {
      return (
        <div className="flex justify-center items-center w-[92px] h-[24px] bg-[#DCFCE7] rounded-[6px]">
          <span className="w-[80px] h-[16px] font-instrument-sans font-semibold text-[14px] leading-[16px] text-center text-[#16A34A]">
            {booking.status}
          </span>
        </div>
      );
    }
    if (column.id === 'pickupTime') {
      return (
        <div className="text-center">
          {booking?.pickupTime?.slice(0, 5)}
        </div>
      );
    }


    // normal fields (API-safe)
    return booking[column.id as keyof Booking] ?? '-';
  };


  const groupBookings = (data: Booking[]): GroupedBooking[] => {
    if (groupedColumns.length === 0) return [];

    const grouped = new Map<string, Booking[]>();

    data.forEach((booking) => {
      const groupKey = groupedColumns
        .map((col) => booking[col as keyof Booking] ?? '-')
        .join('|');

      if (!grouped.has(groupKey)) {
        grouped.set(groupKey, []);
      }
      grouped.get(groupKey)!.push(booking);
    });

    return Array.from(grouped.entries()).map(([key, bookings]) => ({
      groupKey: key,
      // ✅ ONLY VALUES — no column names
      groupValue: key.split('|').join(' / '),
      bookings,
    }));
  };


  const renderRow = (booking: Booking) => (
    <tr
      key={booking.id}
      className="
      bg-white
      hover:bg-[#F8FAFC]
      transition-colors
      border-b border-[#E5E7EB]
    "
    >
      {visibleColumns.map((column) => (
        <td
          key={column.id}
          className={`
          px-4 py-3
          text-[14px]
          leading-[12px]
          text-[#1A1D1F]
          whitespace-nowrap
          ${column.className ?? ''}
        `}
        >
          {renderCell(booking, column)}
        </td>
      ))}
    </tr>
  );

  return (
    <div className="overflow-x-auto bg-white">
      <div className="min-w-full inline-block align-middle">
        <table className="min-w-full">
          <thead className="
                sticky top-0 z-10
                bg-[#F8FAFC]
                border-y border-[#DBEAFE]
              ">

            <tr>
              {visibleColumns?.map((column) => (
                <th
                  key={column.id}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={() => handleDrop(column)}

                  className="
                    py-4 px-4
                    text-[12px]
                    font-bold-[700]
                    leading-[16.5px]
                    text-[#62748E]
                    uppercase
                    tracking-wide
                    whitespace-nowrap
                    select-none
                    cursor-move
                  "
                >
                  <div className="flex items-center gap-2">



                    <span
                      draggable
                      onDragStart={(e) => {
                        // set dragColumn ref
                        dragColumn.current = column;

                        // optional: for native drag-and-drop visuals
                        e.dataTransfer.effectAllowed = "move";
                        e.dataTransfer.setData(
                          'application/json',
                          JSON.stringify({
                            type: 'column',
                            columnId: column.id,
                          })
                        );
                      }}
                      className="cursor-grab select-none"
                    >
                      {column.label}
                    </span>
                    {column.id !== 'actions' && column.id !== 'status' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSort(column.id as string);
                        }}
                        className="opacity-60 hover:opacity-100"
                      >
                        <ArrowUpDown className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </th>

              ))}
            </tr>
            <tr>
              {visibleColumns.map((column) => (
                <th key={column.id}
                  className="py-4 px-4 bg-[#EFF6FF]"
                >
                  <div className="flex items-start">
                    {
                      (column.id !== 'status' && column.id !== 'actions' && column.id !== 'pickupDate' && column.id !== 'pickupTime') &&
                      <input
                        type="text"
                        placeholder={`Search`}
                        value={columnFilters[column.id] || ''}
                        onChange={(e) =>
                          setColumnFilters((prev) => ({
                            ...prev,
                            [column.id]: e.target.value,
                          }))
                        }
                        className={`mt-1 px-2 py-1 h-9 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-[#2C45BE] text-gray-500`}
                      />
                    }
                    {
                      column.id === 'pickupDate' &&
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button
                            variant="outline"
                            className="h-9 px-3 text-sm border-[#E0E0E0] text-[#666666]"
                          >
                            <Calendar className="mr-2 h-4 w-4" />
                            {selectedGridDate
                              ? format(selectedGridDate, 'dd MMMM yyyy')
                              : 'Select'}
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-auto p-0" align="end">
                          <CalendarComponent
                            mode="single"
                            selected={selectedGridDate}
                            onSelect={(date) => {
                              const formatedDate = format(date, 'yyyy-MM-dd');
                              setSelectedGridDate(date);
                              setColumnFilters((prev) => ({
                                ...prev,
                                [column.id]: formatedDate,
                              }))
                            }}
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
                    }
                    {
                      column.id === 'pickupTime' &&
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button
                            variant="outline"
                            className="h-9 px-3 text-sm border-[#E0E0E0] text-[#666666]"
                          >
                            <Clock className="mr-2 h-4 w-4" />
                            {selectedTime || "Select"}
                          </Button>
                        </PopoverTrigger>

                        <PopoverContent className="w-56 p-4" align="end">
                          <input
                            type="time"
                            step="60"
                            value={selectedTime || ""}
                            onChange={(e) => {
                              const timeValue = e.target.value;
                              setSelectedTime(timeValue);

                              setColumnFilters((prev) => ({
                                ...prev,
                                [column.id]: timeValue,
                              }));
                            }}
                            className="w-full px-2 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#2C45BE]"
                          />
                        </PopoverContent>
                      </Popover>

                    }
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groupedColumns?.length > 0
              ? (bookings as GroupedBooking[]).map((group) => (
                <React.Fragment key={group.groupKey}>

                  {/* GROUP HEADER ROW */}
                  <tr
                    className="
                      bg-white
                      border-b border-[#DBEAFE]
                      cursor-pointer
                      hover:bg-[#FAFCFE]
                      transition-colors
                    "
                    onClick={() => toggleGroup(group.groupKey)}
                  >
                    <td colSpan={visibleColumns.length} className="px-6 py-3">
                      <div className="flex items-center gap-3">

                        {/* Chevron */}
                        <div
                          className="
                              w-6 h-6
                              flex items-center justify-center
                              bg-[#EFF6FF]
                              rounded-lg
                              shrink-0
                            "
                        >
                          {expandedGroups.has(group.groupKey) ? (
                            <ChevronDown className="w-4 h-4 text-[#45556C]" />
                          ) : (
                            <ChevronRight className="w-4 h-4 text-[#45556C]" />
                          )}
                        </div>

                        {/* Grouped Value Pills (UNCHANGED) */}
                        <div className="flex items-center gap-3 flex-wrap">
                          {group.groupValue.split(' / ').map((value, idx) => (
                            <div
                              key={idx}
                              className="
                                  h-[31px]
                                  px-[10px]
                                  flex items-center
                                  bg-[#FAFCFE]
                                  border border-[#DBEAFE]
                                  rounded-lg
                                  text-[12px]
                                  font-semibold
                                  text-[#314158]
                                  whitespace-nowrap
                                "
                            >
                              {value}
                            </div>
                          ))}
                        </div>

                      </div>
                    </td>
                  </tr>

                  {/* GROUP ROWS */}
                  {expandedGroups?.has(group.groupKey) &&
                    group?.bookings?.map((booking) => renderRow(booking))}

                </React.Fragment>
              ))
              : (bookings as Booking[])?.map((booking) => renderRow(booking))}
            {
              (bookings as Booking[])?.length === 0 && (
                <tr>
                  <td colSpan={visibleColumns?.length} className="text-center py-10 text-[#45556C]">
                    No bookings found.
                  </td>
                </tr>
              )
            }
          </tbody>

        </table>
      </div>
    </div>
  );
};
