let activeTab = null;
let startTime = null;
const BACKEND_URL = "http://localhost:8000"; // your FastAPI

// When user switches tabs
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  await logCurrentTab();  // log previous tab's time
  const tab = await chrome.tabs.get(activeInfo.tabId);
  activeTab = tab;
  startTime = Date.now();
});

// When tab URL changes
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.active) {
    activeTab = tab;
    startTime = Date.now();
  }
});

async function logCurrentTab() {
  if (!activeTab || !startTime) return;

  const duration = Math.round((Date.now() - startTime) / 1000); // seconds
  if (duration < 3) return; // ignore very short visits

  const data = {
    url: activeTab.url,
    title: activeTab.title,
    duration_seconds: duration,
    timestamp: new Date().toISOString()
  };

  // Send to FastAPI backend
  try {
    await fetch(`${BACKEND_URL}/api/time-log`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
  } catch (e) {
    // Store locally if backend is offline
    chrome.storage.local.get("pending_logs", (result) => {
      const logs = result.pending_logs || [];
      logs.push(data);
      chrome.storage.local.set({ pending_logs: logs });
    });
  }

  startTime = Date.now(); // reset timer
}

// Send any pending logs every 30 seconds
chrome.alarms.create("sync", { periodInMinutes: 0.5 });
chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === "sync") {
    await logCurrentTab();
    syncPendingLogs();
  }
});

async function syncPendingLogs() {
  chrome.storage.local.get("pending_logs", async (result) => {
    const logs = result.pending_logs || [];
    if (logs.length === 0) return;
    try {
      await fetch(`${BACKEND_URL}/api/time-log/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(logs)
      });
      chrome.storage.local.set({ pending_logs: [] });
    } catch (e) {}
  });
}