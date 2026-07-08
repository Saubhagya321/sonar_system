import React, { useState } from "react";
import { X } from "lucide-react";
import { Button } from "../ui/button";
import DragIcon from '@/assets/dragIcon.svg';

const INDENT_PER_LEVEL = 300;
const HORIZONTAL_SEGMENT = 30;
const VERTICAL_SEGMENT = 55;
const ARROW_LENGTH = 30;

interface GroupingBannerProps {
  groupedColumns: string[];
  onRemoveGrouping: (column: string) => void;
  onDropColumn: (columnId: string) => void;
  handleReset: () => void;
}

const formatLabel = (str: string) => {
  if (!str) return "";
  return str
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (s) => s.toUpperCase());
};

export const GroupingBanner: React.FC<GroupingBannerProps> = ({
  groupedColumns,
  onRemoveGrouping,
  onDropColumn,
  handleReset
}) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="p-4 bg-gray-100 rounded-lg"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        const raw = e.dataTransfer.getData('application/json');
        if (!raw) return;

        const data = JSON.parse(raw);
        console.log(data);

        if (data?.type === 'column') {
          onDropColumn(data.columnId);
        }
      }}

    >

      {groupedColumns?.length === 0 && (
        <div className="flex items-center gap-2">
          <img src={DragIcon} alt="Drag Icon" className="w-4 h-4" />
          <span className="text-[14px] leading-[12px] text-[#1A1D1F] select-none">
            Drag a column header here to group by that column
          </span>
        </div>
      )}
      {/* Header */}
      <div className="flex justify-end items-center mb-3 relative" style={{ zIndex: "10" }}>

        <div className="flex gap-2 justify-start">
          {groupedColumns?.length > 0 && (
            <Button
              onClick={() => {
                setExpanded(!expanded);
              }}
              className="flex justify-center items-center px-[16px] py-[10px] w-[85px] h-[32px] bg-[#2C45BE] rounded-[8px] font-instrument-sans font-medium text-[14px] text-white"
            >
              {expanded ? "Collapse All" : "Expand All"}
            </Button>

          )}
          {/* <Button
            onClick={() => handleReset()}
            className="flex justify-center items-center px-[16px] py-[10px] w-[85px] h-[32px] bg-[#2C45BE] rounded-[8px] font-instrument-sans font-medium text-[14px] text-white"
          >
            Reset
          </Button> */}
        </div>
      </div>

      {/* Collapsed View */}
      {!expanded && (
        <div className="flex items-center whitespace-nowrap gap-2" style={{ overflow: "auto" }}>
          {groupedColumns?.map((col, index) => (
            <React.Fragment key={col}>
              <Chip
                label={`${index + 1}. ${formatLabel(col)}`}
                onRemove={() => onRemoveGrouping(col)}
              />
              {index !== groupedColumns?.length - 1 && (
                <Arrow />
              )}
            </React.Fragment>
          ))}
        </div>
      )}

      {/* Expanded Stepper View */}
      {expanded && groupedColumns?.length > 0 && (
        <div className="relative flex flex-col gap-12 mt-4 items-start" style={{ overflow: "auto", paddingTop: "50px" }}>
          {groupedColumns?.map((col, index) => {
            const isLast = index === groupedColumns.length - 1;

            return (
              <div
                key={col}
                className="relative width-min-50-btn"
                style={{ marginLeft: `${index * INDENT_PER_LEVEL}px`, marginTop: '-50px', width: '225px' }}
              >
                <Chip
                  label={`${index + 1}. ${formatLabel(col)}`}
                  onRemove={() => onRemoveGrouping(col)}
                />

                {!isLast && (
                  <svg
                    className="absolute"
                    style={{
                      left: "100%",
                      top: "50%",
                      transform: "translateY(-42%)",
                    }}
                    width={HORIZONTAL_SEGMENT + ARROW_LENGTH + 10}
                    height={VERTICAL_SEGMENT + 10}
                  >
                    {/* Bent Connector Path */}
                    <path
                      d={`
                M 0 ${VERTICAL_SEGMENT / 2}
                H ${HORIZONTAL_SEGMENT}
                V ${VERTICAL_SEGMENT}
                H ${HORIZONTAL_SEGMENT + ARROW_LENGTH}
              `}
                      fill="none"
                      stroke="#2563EB"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />

                    {/* Arrow Head */}
                    <polygon
                      points={`
                ${HORIZONTAL_SEGMENT + ARROW_LENGTH},${VERTICAL_SEGMENT - 6}
                ${HORIZONTAL_SEGMENT + ARROW_LENGTH + 10},${VERTICAL_SEGMENT}
                ${HORIZONTAL_SEGMENT + ARROW_LENGTH},${VERTICAL_SEGMENT + 6}
              `}
                      fill="#2563EB"
                    />
                  </svg>
                )}
              </div>
            );
          })}
        </div>

      )}
    </div>
  );
};

const Chip = ({ label, onRemove }: any) => (
  <div
    className="
      flex items-center gap-2
      px-3 py-1
      text-sm
      rounded-lg
      bg-white
      border border-gray-300
      shadow-sm
      justify-center
    "
  >
    <span className="text-gray-700 font-medium">{label}</span>
    <button onClick={onRemove}>
      <X className="w-4 h-4 text-gray-500 hover:opacity-70" />
    </button>
  </div>
);

const Arrow = () => (
  <div className="flex items-center mx-2">
    <div className="w-6 h-[2px] bg-blue-600" />
    <div
      className="
        w-0 h-0
        border-t-[6px] border-t-transparent
        border-b-[6px] border-b-transparent
        border-l-[8px] border-l-blue-600
      "
    />
  </div>
);
