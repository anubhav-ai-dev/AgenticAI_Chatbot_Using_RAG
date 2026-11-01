const d = require("./ua-file-extract-results-12.json");
const interesting = /^(Theme|Utils|Icons|Workspaces|SettingsView|RadioGroup|UI|Prefs|DocsView|Menu|Motion|CodePreview|AIViz|Notifications|Dialog|ModelSelect)\./;
for (const r of d.results) {
  const seen = new Set();
  for (const c of r.callGraph || []) {
    const cal = String(c.callee).split("\n")[0];
    if (!interesting.test(cal) && !/\.(get|set|init|open|html|escapeHtml|formatTime|formatDateGroup|copyText|setLiveStatus|showToast|resolved|onChange|wire|setUploadBusy|setActive|setLimits|syncModels|setUserId|renderIntegrations|renderGithubRepos|onPane)\b/.test(cal)) continue;
    if (!/^(Theme|Utils|Icons|Workspaces|SettingsView|RadioGroup|UI|Prefs|DocsView|Notifications|Dialog)\./.test(cal)) continue;
    const key = c.caller + " -> " + cal;
    if (seen.has(key)) continue;
    seen.add(key);
    console.log(r.path + " | " + c.caller + " -> " + cal + " @" + c.lineNumber);
  }
}
