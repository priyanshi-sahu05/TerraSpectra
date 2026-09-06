function Timeline({ dates, selectedDate, onDateChange }) {
  const selectedIndex = dates.indexOf(selectedDate);

  return (
    <div className="timeline">
      <h3>Historical Risk Timeline</h3>

      <input
        type="range"
        min="0"
        max={dates.length - 1}
        value={selectedIndex}
        onChange={(event) =>
          onDateChange(dates[Number(event.target.value)])
        }
      />

      <div className="timeline-labels">
        {dates.map((date) => (
          <span key={date}>{date}</span>
        ))}
      </div>

      <p>
        Selected Date: <strong>{selectedDate}</strong>
      </p>
    </div>
  );
}

export default Timeline;