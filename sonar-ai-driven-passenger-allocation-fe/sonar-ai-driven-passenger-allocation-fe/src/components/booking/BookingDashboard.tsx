import React from 'react';
import { Header } from './Header';
import { PageTitleSection } from './PageTitleSection';
import { GroupingBanner } from './GroupingBanner';
import { BookingTable } from './BookingTable';
import { PaginationControls } from './PaginationControls';
import { RatingModal } from './RatingModal';
import { RejectModal } from './RejectModal';
import { mockBookings } from '@/data/mockBookings';
import { fetchPendingBookings } from '@/services/booking.service';
import { Booking } from '@/types/booking';
import { format } from 'date-fns';
import { toast } from 'react-toastify';
import axios from 'axios';

export const BookingDashboard: React.FC = () => {
  // const [bookings, setBookings] = React.useState<Booking[]>(mockBookings);
  const [bookings, setBookings] = React.useState<Booking[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const [selectedDate, setSelectedDate] = React.useState<Date | undefined>();
  const [groupedColumns, setGroupedColumns] = React.useState<string[]>([]);
  const [sortColumn, setSortColumn] = React.useState<string | null>(null);
  const [sortDirection, setSortDirection] = React.useState<'asc' | 'desc'>('asc');
  const [currentPage, setCurrentPage] = React.useState(1);
  const [rowsPerPage, setRowsPerPage] = React.useState(15);
  const [isRatingModalOpen, setIsRatingModalOpen] = React.useState(false);
  const [isRejectModalOpen, setIsRejectModalOpen] = React.useState(false);
  const [selectedBookingId, setSelectedBookingId] = React.useState<string | null>(null);
  const [selectedBooking, setSelectedBooking] = React.useState<Booking | null>(null);

  const [columnFilters, setColumnFilters] = React.useState<Record<string, string>>({});
  const [selectedTime, setSelectedTime] = React.useState<string>('');
  const [selectedGridDate, setSelectedGridDate] = React.useState<Date | undefined>();

  React.useEffect(() => {
    loadBookings();
  }, []);

  React.useEffect(() => {
    setCurrentPage(1);
  }, [selectedDate]);


  // Filter
  const filteredBookings = React.useMemo(() => {
    let data = bookings;

    // Date filter
    if (selectedDate) {
      const dateStr = format(selectedDate, 'yyyy-MM-dd');
      data = data?.filter((b) => b?.pickupDate === dateStr);
    }

    // Column-based filtering
    Object?.entries(columnFilters)?.forEach(([column, value]) => {
      if (value?.trim() !== '') {
        data = data?.filter((booking) => {
          const cellValue = String(booking[column as keyof Booking] ?? '');
          // 🎯 Exact match only for pickupTime
          if (column === 'pickupTime') {
            return cellValue?.slice(0, 5) === value;
            // Handles "08:00:00" vs "08:00"
          }
          // 🔎 Default partial match for other columns
          return cellValue?.toLowerCase()?.includes(value?.toLowerCase());
        });
      }
    });


    return data;
  }, [bookings, selectedDate, columnFilters]);


  // Sort
  const sortedBookings = React.useMemo(() => {
    if (!sortColumn) return filteredBookings;
    return [...filteredBookings].sort((a, b) => {
      const aVal = a[sortColumn as keyof Booking];
      const bVal = b[sortColumn as keyof Booking];

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
  }, [filteredBookings, sortColumn, sortDirection]);


  // Paginate bookings
  // const paginatedBookings = React.useMemo(() => {
  //   const startIndex = (currentPage - 1) * rowsPerPage;
  //   const endIndex = startIndex + rowsPerPage;
  //   return sortedBookings.slice(startIndex, endIndex);
  // }, [sortedBookings, currentPage, rowsPerPage]);

  const groupedData = React.useMemo(() => {
    if (groupedColumns.length === 0) return [];

    const map = new Map<string, Booking[]>();

    sortedBookings.forEach((booking) => {
      const key = groupedColumns
        .map((col) => booking[col as keyof Booking] ?? '-')
        .join('|');

      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(booking);
    });

    return Array.from(map.entries()).map(([key, bookings]) => ({
      groupKey: key,
      groupValue: key.split('|').join(' / '),
      bookings,
    }));
  }, [sortedBookings, groupedColumns]);

  const paginatedGroupedData = React.useMemo(() => {
    if (groupedColumns.length === 0) return [];

    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    return groupedData.slice(start, end);
  }, [groupedData, currentPage, rowsPerPage]);

  const paginatedBookings = React.useMemo(() => {
    if (groupedColumns.length > 0) return [];
    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    return sortedBookings.slice(start, end);
  }, [sortedBookings, currentPage, rowsPerPage, groupedColumns]);

  // const totalPages = Math.ceil(sortedBookings.length / rowsPerPage);
  const totalPages = React.useMemo(() => {
    if (groupedColumns.length > 0) {
      return Math.ceil(groupedData.length / rowsPerPage);
    }
    return Math.ceil(sortedBookings.length / rowsPerPage);
  }, [groupedColumns, groupedData.length, sortedBookings.length, rowsPerPage]);


  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const handleDragStart = (e: React.DragEvent, columnId: string) => {
    e.dataTransfer.setData('columnId', columnId);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  /**
   * Function to handle dropping a column header for grouping
   * @param columnId Column Name
   * @returns 
   */
  const handleDropColumn = (columnId: string) => {
    if (columnId === "status" || columnId === "actions") return;
    if (!groupedColumns.includes(columnId)) {
      setGroupedColumns((prev) => [...prev, columnId]);
    }
  };

  const handleReset = () => {
    setColumnFilters({});
    setGroupedColumns([]);
    setSortColumn(null);
    setSortDirection('asc');

    setSelectedDate(undefined);   // Clear date
    setSelectedGridDate(undefined);   // Clear grid date
    setSelectedTime(undefined);   // Clear grid time
    setCurrentPage(1);            // Reset pagination

    // get data from backend again to reset any status changes
    loadBookings();
  }

  const handleRemoveGrouping = (column: string) => {
    setGroupedColumns(groupedColumns.filter((col) => col !== column));
  };

  const handleAccept = (booking: Booking) => {
    setSelectedBooking(booking);
    setIsRatingModalOpen(true);
  };

  const handleReject = (booking: Booking) => {
    setSelectedBooking(booking);
    setIsRejectModalOpen(true);
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handleRowsPerPageChange = (rows: number) => {
    setRowsPerPage(rows);
    setCurrentPage(1);
  };
  const loadBookings = React.useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchPendingBookings();
      setBookings(data);
    } catch (err) {
      console.error(err);
      setError('Failed to load bookings');
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-[#F5F5F5]">
      <Header />

      <main className="pt-24 px-10 pb-10">
        <div className="bg-white border border-[#E6E6E6] rounded-md">
          <div className="px-4 pt-4">
            <PageTitleSection
              totalResults={filteredBookings.length}
              selectedDate={selectedDate}
              onDateChange={setSelectedDate}
              handleReset={handleReset}
            />
          </div>

          {/* <div onDragOver={handleDragOver} onDrop={handleDrop} className="px-4"> */}
          <GroupingBanner
            groupedColumns={groupedColumns}
            onRemoveGrouping={handleRemoveGrouping}
            onDropColumn={handleDropColumn}
            handleReset={handleReset}
          />
          {loading ? (
            <div className="flex justify-center items-center py-20">
              <div className="h-8 w-8 border-4 border-[#2C45BE] border-t-transparent rounded-full animate-spin" />
              <span className="ml-3 text-sm text-[#62748E]">
                Loading bookings...
              </span>
            </div>
          ) : error ? (
            <div className="p-6 text-center text-sm text-red-500">
              {error}
            </div>
          ) : (
            <BookingTable
              bookings={groupedColumns.length > 0 ? paginatedGroupedData : paginatedBookings}
              groupedColumns={groupedColumns}
              sortColumn={sortColumn}
              sortDirection={sortDirection}
              onSort={handleSort}
              onDragStart={handleDragStart}
              onAccept={handleAccept}
              onReject={handleReject}
              columnFilters={columnFilters}
              setColumnFilters={setColumnFilters}
              selectedGridDate={selectedGridDate}
              setSelectedGridDate={setSelectedGridDate}
              selectedTime={selectedTime}
              setSelectedTime={setSelectedTime}
            />
          )}

        </div>


        <PaginationControls
          currentPage={currentPage}
          totalPages={totalPages}
          rowsPerPage={rowsPerPage}
          onPageChange={handlePageChange}
          onRowsPerPageChange={handleRowsPerPageChange}
        />
      </main>

      <RatingModal
        isOpen={isRatingModalOpen}
        booking={selectedBooking}
        onClose={() => {
          setIsRatingModalOpen(false);
          setSelectedBooking(null);
        }}
        onSuccess={loadBookings}
      />


      <RejectModal
        isOpen={isRejectModalOpen}
        booking={selectedBooking}
        onClose={() => {
          setIsRejectModalOpen(false);
          setSelectedBooking(null);
        }}
        onSuccess={loadBookings}
      />

    </div>
  );
};
