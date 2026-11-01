#!/usr/bin/env node
"use strict";

const fs = require("fs");

const DIR_PATTERNS = [
  [/^(routes|api|controllers|endpoints|handlers|serializers|controller|routers|blueprints)$/i, "api"],
  [/^(services|core|lib|domain|logic|signals|internal|composables|mailers|jobs|channels)$/i, "service"],
  [/^(models|db|data|persistence|repository|entities|migrations|entity|sql|database|schema)$/i, "data"],
  [/^(components|views|pages|ui|layouts|screens)$/i, "ui"],
  [/^(middleware|plugins|interceptors|guards)$/i, "middleware"],
  [/^(utils|helpers|common|shared|tools|templatetags|pkg)$/i, "utility"],
  [/^(config|constants|env|settings|management|commands)$/i, "config"],
  [/^(__tests__|test|tests|spec|specs)$/i, "test"],
  [/^(types|interfaces|schemas|contracts|dtos|dto|request|response)$/i, "types"],
  [/^hooks$/i, "hooks"],
  [/^(store|state|reducers|actions|slices)$/i, "state"],
  [/^(assets|static|public)$/i, "assets"],
  [/^(cmd|bin)$/i, "entry"],
  [/^(docs|documentation|wiki)$/i, "documentation"],
  [/^(deploy|deployment|infra|infrastructure|k8s|kubernetes|helm|charts|terraform|tf|docker)$/i, "infrastructure"],
  [/^\.(github|gitlab|circleci)$/i, "ci-cd"],
];

function splitPath(p) {
  return String(p || "").replace(/\\/g, "/").split("/").filter(Boolean);
}

function commonDirPrefix(paths) {
  if (!paths.length) return [];
  const segs = paths.map(splitPath);
  const prefix = [];
  const shortest = Math.min(...segs.map((s) => s.length));
  for (let i = 0; i < shortest; i++) {
    const seg = segs[0][i];
    const allShareDir = segs.every((s) => s[i] === seg && s.length > i + 1);
    if (!allShareDir) break;
    prefix.push(seg);
  }
  return prefix;
}

function groupOf(filePath, prefixLen) {
  const segs = splitPath(filePath);
  const rest = segs.slice(prefixLen);
  if (rest.length <= 1) return "root";
  return rest[0];
}

function nestedGroupOf(filePath, prefixLen) {
  const segs = splitPath(filePath);
  const rest = segs.slice(prefixLen);
  if (rest.length <= 1) return "root";
  if (rest.length === 2) return rest[0] + "/(root)";
  return rest[0] + "/" + rest[1];
}

function matchDirPattern(name) {
  if (!name || name === "root") return null;
  const base = name.split("/").pop();
  for (const [re, label] of DIR_PATTERNS) {
    if (re.test(base)) return label;
  }
  if (/^src\/main\/java$/i.test(name)) return "service";
  if (/^src\/test\/java$/i.test(name)) return "test";
  return null;
}

function filePatternLabel(filePath, name) {
  const p = String(filePath || "").replace(/\\/g, "/");
  const base = name || p.split("/").pop() || "";
  if (/\.test\.[^.]+$|\.spec\.[^.]+$/.test(base)) return "test";
  if (/^test_.*\.py$|_test\.go$|Test\.java$|_spec\.rb$|Test\.php$|Tests\.cs$/.test(base)) return "test";
  if (/\.d\.ts$/.test(base)) return "types";
  if (/^(index\.(ts|js)|__init__\.py)$/.test(base)) return "entry";
  if (base === "manage.py" && !p.includes("/")) return "entry";
  if (base === "wsgi.py" || base === "asgi.py") return "config";
  if (base === "main.go" && /(?:^|\/)cmd\/[^/]+\/main\.go$/.test(p)) return "entry";
  if ((base === "main.rs" || base === "lib.rs") && /(?:^|\/)src\/(main|lib)\.rs$/.test(p)) return "entry";
  if (base === "Application.java" || base === "Program.cs") return "entry";
  if (base === "config.ru") return "entry";
  if (/^(Cargo\.toml|go\.mod|Gemfile|pom\.xml|build\.gradle|composer\.json|Pipfile|pyproject\.toml|requirements\.txt)$/.test(base)) return "config";
  if (/^Dockerfile/.test(base) || /^docker-compose\./.test(base)) return "infrastructure";
  if (/\.tf$|\.tfvars$/.test(base)) return "infrastructure";
  if (/(^|\/)\.github\/workflows\//.test(p) || base === ".gitlab-ci.yml" || base === "Jenkinsfile") return "ci-cd";
  if (/\.sql$/.test(base)) return "data";
  if (/\.(graphql|gql|proto)$/.test(base)) return "types";
  if (/\.(md|rst)$/.test(base)) return "documentation";
  if (base === "Makefile") return "infrastructure";
  if (base === "Procfile") return "infrastructure";
  if (base === "devcontainer.json") return "infrastructure";
  return null;
}

function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!inputPath || !outputPath) {
    console.error("Usage: node ua-arch-analyze.js <input.json> <output.json>");
    process.exit(1);
  }

  let data;
  try {
    data = JSON.parse(fs.readFileSync(inputPath, "utf8"));
  } catch (err) {
    console.error("Failed to read input: " + err.message);
    process.exit(1);
  }

  const fileNodes = data.fileNodes || [];
  const importEdges = data.importEdges || [];
  const allEdges = data.allEdges || [];

  const paths = fileNodes.map((n) => n.filePath || n.name || "");
  const prefix = commonDirPrefix(paths);
  const prefixLen = prefix.length;

  const directoryGroups = {};
  const nestedGroups = {};
  const nodeTypeGroups = {};
  const idToNode = {};

  for (const node of fileNodes) {
    idToNode[node.id] = node;
    const g = groupOf(node.filePath, prefixLen);
    const ng = nestedGroupOf(node.filePath, prefixLen);
    (directoryGroups[g] || (directoryGroups[g] = [])).push(node.id);
    (nestedGroups[ng] || (nestedGroups[ng] = [])).push(node.id);
    const t = node.type || "file";
    (nodeTypeGroups[t] || (nodeTypeGroups[t] = [])).push(node.id);
  }

  for (const key of Object.keys(directoryGroups)) directoryGroups[key].sort();
  for (const key of Object.keys(nestedGroups)) nestedGroups[key].sort();
  for (const key of Object.keys(nodeTypeGroups)) nodeTypeGroups[key].sort();

  function groupForId(id) {
    const node = idToNode[id];
    if (!node) return "root";
    return groupOf(node.filePath, prefixLen);
  }

  const fanOut = {};
  const fanIn = {};
  for (const node of fileNodes) {
    fanOut[node.id] = 0;
    fanIn[node.id] = 0;
  }

  const interCounts = {};
  for (const e of importEdges) {
    if (fanOut[e.source] == null) fanOut[e.source] = 0;
    if (fanIn[e.target] == null) fanIn[e.target] = 0;
    fanOut[e.source] += 1;
    fanIn[e.target] += 1;
    const from = groupForId(e.source);
    const to = groupForId(e.target);
    if (from === to) continue;
    const k = from + "\t" + to;
    interCounts[k] = (interCounts[k] || 0) + 1;
  }

  const interGroupImports = Object.keys(interCounts)
    .map((k) => {
      const [from, to] = k.split("\t");
      return { from, to, count: interCounts[k] };
    })
    .sort((a, b) => b.count - a.count || a.from.localeCompare(b.from));

  const intraGroupDensity = {};
  for (const g of Object.keys(directoryGroups)) {
    let internalEdges = 0;
    let totalEdges = 0;
    for (const e of importEdges) {
      const from = groupForId(e.source);
      const to = groupForId(e.target);
      if (from === g || to === g) {
        totalEdges += 1;
        if (from === g && to === g) internalEdges += 1;
      }
    }
    intraGroupDensity[g] = {
      internalEdges,
      totalEdges,
      density: totalEdges === 0 ? 0 : Number((internalEdges / totalEdges).toFixed(4)),
    };
  }

  const patternMatches = {};
  for (const g of Object.keys(directoryGroups)) {
    const label = matchDirPattern(g);
    if (label) patternMatches[g] = label;
  }

  const filePatternMatches = {};
  for (const node of fileNodes) {
    const label = filePatternLabel(node.filePath, node.name);
    if (label) filePatternMatches[node.id] = label;
  }

  const crossMap = {};
  const nonCodeLinks = [];
  for (const e of allEdges) {
    const sType = (idToNode[e.source] && idToNode[e.source].type) || "unknown";
    const tType = (idToNode[e.target] && idToNode[e.target].type) || "unknown";
    const k = sType + "\t" + tType + "\t" + (e.type || "unknown");
    crossMap[k] = (crossMap[k] || 0) + 1;
    const codeTypes = new Set(["file", "endpoint"]);
    if (!codeTypes.has(sType) || !codeTypes.has(tType)) {
      nonCodeLinks.push({
        source: e.source,
        target: e.target,
        edgeType: e.type,
        fromType: sType,
        toType: tType,
      });
    }
  }
  const crossCategoryEdges = Object.keys(crossMap)
    .map((k) => {
      const [fromType, toType, edgeType] = k.split("\t");
      return { fromType, toType, edgeType, count: crossMap[k] };
    })
    .sort((a, b) => b.count - a.count || a.fromType.localeCompare(b.fromType));

  const pair = {};
  for (const row of interGroupImports) {
    const k = [row.from, row.to].sort().join("\t");
    if (!pair[k]) pair[k] = {};
    pair[k][row.from + "->" + row.to] = row.count;
  }
  const dependencyDirection = [];
  for (const row of interGroupImports) {
    const reverse = interGroupImports.find((r) => r.from === row.to && r.to === row.from);
    const revCount = reverse ? reverse.count : 0;
    if (row.count > revCount) {
      dependencyDirection.push({ dependent: row.from, dependsOn: row.to });
    }
  }

  const infraFiles = [];
  let hasDockerfile = false;
  let hasCompose = false;
  let hasK8s = false;
  let hasTerraform = false;
  let hasCI = false;
  for (const node of fileNodes) {
    const p = String(node.filePath || "").replace(/\\/g, "/");
    const base = node.name || p.split("/").pop() || "";
    const isDocker = /^Dockerfile/.test(base);
    const isCompose = /^docker-compose\./.test(base);
    const isK8s = /(^|\/)(k8s|kubernetes|helm|charts)\//.test(p) || /\.ya?ml$/.test(base) && /(k8s|kubernetes|helm)/.test(p);
    const isTf = /\.tf$|\.tfvars$/.test(base) || /(^|\/)(terraform|tf)\//.test(p);
    const isCI = /(^|\/)\.github\/workflows\//.test(p) || base === ".gitlab-ci.yml" || base === "Jenkinsfile" || /(^|\/)\.circleci\//.test(p);
    const isProc = base === "Procfile" || base === "Makefile" || base === "devcontainer.json";
    if (isDocker) hasDockerfile = true;
    if (isCompose) hasCompose = true;
    if (isK8s) hasK8s = true;
    if (isTf) hasTerraform = true;
    if (isCI) hasCI = true;
    if (isDocker || isCompose || isK8s || isTf || isCI || isProc) infraFiles.push(p);
  }

  const schemaFiles = [];
  const migrationFiles = [];
  const dataModelFiles = [];
  const apiHandlerFiles = [];
  for (const node of fileNodes) {
    const p = String(node.filePath || "").replace(/\\/g, "/");
    const base = node.name || "";
    const tags = node.tags || [];
    if (/\.(sql|graphql|gql|proto|prisma)$/.test(base) || node.type === "schema") schemaFiles.push(p);
    if (/migrations?\//.test(p)) migrationFiles.push(p);
    if (
      tags.includes("data-model") ||
      /(^|\/)(models|entities|entity)\//.test(p) ||
      /(^|\/)database\.py$/.test(p) ||
      base === "models.py"
    ) {
      if (node.type === "file") dataModelFiles.push(p);
    }
    if (tags.includes("api-handler") && node.type === "file") apiHandlerFiles.push(p);
    if (node.type === "endpoint") {
      const parent = p;
      if (!apiHandlerFiles.includes(parent)) apiHandlerFiles.push(parent);
    }
  }

  const docGroups = new Set();
  for (const node of fileNodes) {
    const p = String(node.filePath || "").replace(/\\/g, "/");
    if (node.type === "document" || /\.(md|rst)$/.test(p)) {
      docGroups.add(groupOf(node.filePath, prefixLen));
    }
  }
  for (const e of allEdges) {
    if (e.type === "documents" && idToNode[e.target]) {
      docGroups.add(groupOf(idToNode[e.target].filePath, prefixLen));
    }
  }
  const allGroups = Object.keys(directoryGroups).sort();
  const undocumentedGroups = allGroups.filter((g) => !docGroups.has(g));

  const filesPerGroup = {};
  for (const g of allGroups) filesPerGroup[g] = directoryGroups[g].length;
  const nodeTypeCounts = {};
  for (const t of Object.keys(nodeTypeGroups)) nodeTypeCounts[t] = nodeTypeGroups[t].length;

  const result = {
    scriptCompleted: true,
    commonPrefix: prefix.join("/"),
    directoryGroups,
    nestedGroups,
    nodeTypeGroups,
    crossCategoryEdges,
    nonCodeLinks,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    filePatternMatches,
    deploymentTopology: {
      hasDockerfile,
      hasCompose,
      hasK8s,
      hasTerraform,
      hasCI,
      infraFiles,
    },
    dataPipeline: {
      schemaFiles,
      migrationFiles,
      dataModelFiles: [...new Set(dataModelFiles)],
      apiHandlerFiles: [...new Set(apiHandlerFiles)],
    },
    docCoverage: {
      groupsWithDocs: docGroups.size,
      totalGroups: allGroups.length,
      coverageRatio: allGroups.length === 0 ? 0 : Number((docGroups.size / allGroups.length).toFixed(4)),
      undocumentedGroups,
    },
    dependencyDirection,
    fileStats: {
      totalFileNodes: fileNodes.length,
      filesPerGroup,
      nodeTypeCounts,
    },
    fileFanIn: fanIn,
    fileFanOut: fanOut,
  };

  try {
    fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
  } catch (err) {
    console.error("Failed to write output: " + err.message);
    process.exit(1);
  }
}

main();
