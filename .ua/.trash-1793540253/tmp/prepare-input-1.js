const fs = require("fs");
const root = "F:/Personal Projects/AgenticAI_Chatbot_Using_RAG";
const p = JSON.parse(fs.readFileSync(root + "/.ua/tmp/batch-prompt-1.json", "utf8"));
const input = {
  projectRoot: "F:\\Personal Projects\\AgenticAI_Chatbot_Using_RAG",
  batchFiles: p.files,
  batchImportData: p.batchImportData,
};
fs.writeFileSync(root + "/.ua/tmp/ua-file-analyzer-input-1.json", JSON.stringify(input, null, 2));
console.log("files", input.batchFiles.length, "imports", Object.keys(input.batchImportData).length);
