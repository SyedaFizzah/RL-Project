chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  if (tabs[0]) {
    const url = new URL(tabs[0].url);
    document.getElementById("current-tab").textContent = url.hostname;
  }
});

chrome.storage.local.get("pending_logs", (result) => {
  const logs = result.pending_logs || [];
  document.getElementById("pending").textContent = logs.length;
});

// Fetch today's total from backend
fetch("http://localhost:8000/api/time-log/today")
  .then(r => r.json())
  .then(data => {
    const mins = Math.round(data.total_seconds / 60);
    document.getElementById("total-time").textContent = `${mins} min`;
  })
  .catch(() => {
    document.getElementById("total-time").textContent = "Offline";
  });

document.getElementById("sync-btn").addEventListener("click", () => {
  chrome.runtime.sendMessage({ action: "sync" });
  window.close();
});