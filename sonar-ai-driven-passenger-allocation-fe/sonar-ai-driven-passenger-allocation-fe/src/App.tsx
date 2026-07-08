import { Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import { BookingDashboard } from "./components/booking/BookingDashboard";
import { ToastContainer } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

function App() {
  return (
    <Suspense fallback={<p>Loading...</p>}>
      <>
        <Routes>
          <Route path="/" element={<BookingDashboard />} />
        </Routes>

        {/* Toast Container */}
        <ToastContainer
          position="top-right"
          autoClose={3000}          // auto close after 3s
          hideProgressBar={false}
          newestOnTop={true}       // new toast appears on top
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="colored"
          toastClassName="!rounded-lg !font-medium !text-sm !px-5 !py-3"
          bodyClassName="!flex !items-center"
        />
      </>
    </Suspense>
  );
}

export default App;
