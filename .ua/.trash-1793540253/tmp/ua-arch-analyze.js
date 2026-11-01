#!/usr/bin/env node
"use strict";

const fs = require("fs");

const DIR_PATTERNS = [
  [new Set(["routes", "api", "controllers", "endpoints", "handlers", "routers", "blueprints", "serializers", "controller"]), "api"],
  [new Set(["services", "core", "lib", "domain", "logic", "internal", "signals", "composables", "mailers", "jobs", "channels"]), "service"],
  [new Set(["models", "db", "data", "persistence", "repository", "entities", "migrations", "entity", "sql", "database", "schema"]), "data"],
  [new Set(["components", "views", "pages", "ui", "layouts", "screens"]), "ui"],
  [new Set(["middleware", "plugins", "interceptors", "guards"]), "middleware"],
  [new Set(["utils", "helpers", "common", "shared", "tools", "templatetags", "pkg"]), "utility"],
  [new Set(["config", "constants", "env", "settings", "management", "commands"]), "config"],
  [new Set(["__tests__", "test", "tests", "spec", "specs"]), "test"],
  [new Set(["types", "interfaces", "schemas", "contracts", "dtos", "dto", "request", "response"]), "types"],
  [new Set(["hooks"]), "hooks"],
  [new Set(["store", "state", "reducers", "actions", "slices"]), "state"],
  [new Set(["assets", "static", "public"]), "assets"],
  [new Set(["cmd", "bin"]), "entry"],
  [new Set(["docs", "documentation", "wiki"]), "documentation"],
  [new Set(["deploy", "deployment", "infra", "infrastructure", "k8s", "kubernetes", "helm", "charts", "terraform", "tf", "docker"]), "infrastructure"],
  [new Set([".github", ".gitlab", ".circleci"]), "ci-cd"],
];

function norm(p) {
  return String(p || "").replace(/\\/g, "/").replace(/^\.\//, "");
}

function basename(p) {
  const n = norm(p);
  const i = n.lastIndexOf("/");
  return i === -1 ? n : n.slice(i + 1);
}

function commonDirPrefix(paths) {
  const splits = paths.map((p) => norm(p).split("/").filter(Boolean));
  if (splits.length === 0) return "";
  const first = splits[0];
  let i = 0;
  while (i < first.length) {
    const seg = first[i];
    const ok = splits.every((parts) => parts.length > i + 1 && parts[i] === seg);
    if (!ok) break;
    i++;
  }
  return i === 0 ? "" : first.slice(0, i).join("/") + "/";
}

function groupOf(filePath, prefix) {
  let rel = norm(filePath);
  if (prefix && rel.startsWith(prefix)) rel = rel.slice(prefix.length);
  const parts = rel.split("/").filter(Boolean);
  if (parts.length <= 1) return "root";
  return parts[0];
}

function packageOf(filePath) {
  const parts = norm(filePath).split("/").filter(Boolean);
  if (parts.length <= 1) return "root";
  if (parts.length === 2) return parts[0];
  return parts[0] + "/" + parts[1];
}

function matchDirPattern(name) {
  const key = String(name || "").toLowerCase();
  for (const [set, label] of DIR_PATTERNS) {
    if (set.has(key)) return label;
  }
  if (key === "src/main/java") return "service";
  if (key === "src/test/java") return "test";
  return null;
}

function matchFilePattern(filePath) {
  const p = norm(filePath);
  const base = basename(p);
  const lower = base.toLowerCase();

  if (/(^|\/)Dockerfile$/i.test(p) || /^Dockerfile/i.test(base)) return "infrastructure";
  if (/^docker-compose\./i.test(base) || lower === "docker-compose.yml" || lower === "docker-compose.yaml") return "infrastructure";
  if (/\.tf$|\.tfvars$/i.test(base)) return "infrastructure";
  if (/(^|\/)\.github\/workflows\//i.test(p) || lower === ".gitlab-ci.yml" || lower === "jenkinsfile") return "ci-cd";
  if (/\.sql$/i.test(base)) return "data";
  if (/\.(graphql|gql|proto)$/i.test(base)) return "types";
  if (/\.(md|rst)$/i.test(base)) return "documentation";
  if (lower === "makefile") return "infrastructure";

  if (
    /\.test\.[^.]+$/i.test(base) ||
    /\.spec\.[^.]+$/i.test(base) ||
    /^test_.+\.py$/i.test(base) ||
    /_test\.go$/i.test(base) ||
    /Test\.java$/i.test(base) ||
    /_spec\.rb$/i.test(base) ||
    /Test\.php$/i.test(base) ||
    /Tests\.cs$/i.test(base)
  ) {
    return "test";
  }
  if (/\.d\.ts$/i.test(base)) return "types";

  if ((lower === "index.ts" || lower === "index.js" || lower === "__init__.py") && p.split("/").length <= 2) return "entry";
  if (lower === "manage.py" && !p.includes("/")) return "entry";
  if (lower === "wsgi.py" || lower === "asgi.py") return "config";
  if (lower === "main.go" && /^cmd\/[^/]+\/main\.go$/.test(p)) return "entry";
  if ((lower === "main.rs" || lower === "lib.rs") && p.split("/").length === 2 && p.startsWith("src/")) return "entry";
  if (lower === "application.java" || lower === "program.cs") return "entry";
  if (lower === "config.ru") return "entry";
  if (["cargo.toml", "go.mod", "gemfile", "pom.xml", "build.gradle", "composer.json", "pipfile", "pyproject.toml", "package.json"].includes(lower)) return "config";
  if (lower === "conftest.py") return "test";
  if (lower === "main.py" || lower === "app.py") return "entry";
  return null;
}

function nodeTypeOf(id, byId) {
  const n = byId.get(id);
  return n ? n.type : "unknown";
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
    console.error("Failed to read input:", err.message);
    process.exit(1);
  }

  const fileNodes = data.fileNodes || [];
  const importEdges = data.importEdges || [];
  const allEdges = data.allEdges || [];
  const byId = new Map(fileNodes.map((n) => [n.id, n]));

  const paths = fileNodes.map((n) => norm(n.filePath || n.name || n.id));
  const prefix = commonDirPrefix(paths);

  const directoryGroups = {};
  const packageGroups = {};
  for (const n of fileNodes) {
    const fp = norm(n.filePath || n.name || "");
    const g = groupOf(fp, prefix);
    const pkg = packageOf(fp);
    (directoryGroups[g] || (directoryGroups[g] = [])).push(n.id);
    (packageGroups[pkg] || (packageGroups[pkg] = [])).push(n.id);
  }

  const nodeTypeGroups = {};
  for (const n of fileNodes) {
    const t = n.type || "file";
    (nodeTypeGroups[t] || (nodeTypeGroups[t] = [])).push(n.id);
  }

  const idToGroup = new Map();
  for (const [g, ids] of Object.entries(directoryGroups)) {
    for (const id of ids) idToGroup.set(id, g);
  }
  const idToPkg = new Map();
  for (const [g, ids] of Object.entries(packageGroups)) {
    for (const id of ids) idToPkg.set(id, g);
  }

  const fanOut = {};
  const fanIn = {};
  for (const n of fileNodes) {
    fanOut[n.id] = 0;
    fanIn[n.id] = 0;
  }
  const adjacency = {};
  for (const e of importEdges) {
    if (!adjacency[e.source]) adjacency[e.source] = [];
    adjacency[e.source].push(e.target);
    if (fanOut[e.source] !== undefined) fanOut[e.source] += 1;
    else fanOut[e.source] = 1;
    if (fanIn[e.target] !== undefined) fanIn[e.target] += 1;
    else fanIn[e.target] = 1;
  }

  const groupImportsFrom = {};
  const groupImportedBy = {};
  const pairCounts = {};
  for (const g of Object.keys(directoryGroups)) {
    groupImportsFrom[g] = {};
    groupImportedBy[g] = {};
  }

  function bump(map, key) {
    map[key] = (map[key] || 0) + 1;
  }

  for (const e of importEdges) {
    const from = idToGroup.get(e.source) || "unknown";
    const to = idToGroup.get(e.target) || "unknown";
    if (!groupImportsFrom[from]) groupImportsFrom[from] = {};
    if (!groupImportedBy[to]) groupImportedBy[to] = {};
    if (from !== to) {
      bump(groupImportsFrom[from], to);
      bump(groupImportedBy[to], from);
    }
    const k = from + "\0" + to;
    pairCounts[k] = (pairCounts[k] || 0) + 1;
  }

  const interGroupImports = Object.entries(pairCounts)
    .map(([k, count]) => {
      const [from, to] = k.split("\0");
      return { from, to, count };
    })
    .filter((x) => x.from !== x.to)
    .sort((a, b) => b.count - a.count || a.from.localeCompare(b.from));

  const pkgPair = {};
  for (const e of importEdges) {
    const from = idToPkg.get(e.source) || "unknown";
    const to = idToPkg.get(e.target) || "unknown";
    if (from === to) continue;
    const k = from + "\0" + to;
    pkgPair[k] = (pkgPair[k] || 0) + 1;
  }
  const interPackageImports = Object.entries(pkgPair)
    .map(([k, count]) => {
      const [from, to] = k.split("\0");
      return { from, to, count };
    })
    .sort((a, b) => b.count - a.count || a.from.localeCompare(b.from));

  const intraGroupDensity = {};
  for (const g of Object.keys(directoryGroups)) {
    let internalEdges = 0;
    let totalEdges = 0;
    for (const e of importEdges) {
      const from = idToGroup.get(e.source);
      const to = idToGroup.get(e.target);
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
  for (const n of fileNodes) {
    const label = matchFilePattern(n.filePath || "");
    if (label) filePatternMatches[n.id] = label;
  }

  const crossMap = {};
  for (const e of allEdges) {
    const fromType = nodeTypeOf(e.source, byId);
    const toType = nodeTypeOf(e.target, byId);
    const edgeType = e.type || "unknown";
    const k = fromType + "\0" + toType + "\0" + edgeType;
    crossMap[k] = (crossMap[k] || 0) + 1;
  }
  const crossCategoryEdges = Object.entries(crossMap)
    .map(([k, count]) => {
      const [fromType, toType, edgeType] = k.split("\0");
      return { fromType, toType, edgeType, count };
    })
    .sort((a, b) => b.count - a.count || a.fromType.localeCompare(b.fromType));

  const nonCodeLinks = [];
  for (const e of allEdges) {
    const fromType = nodeTypeOf(e.source, byId);
    const toType = nodeTypeOf(e.target, byId);
    const codeTypes = new Set(["file"]);
    const nonCode = new Set(["config", "document", "service", "pipeline", "table", "schema", "resource", "endpoint"]);
    if ((nonCode.has(fromType) && codeTypes.has(toType)) || (nonCode.has(toType) && codeTypes.has(fromType))) {
      nonCodeLinks.push({
        source: e.source,
        target: e.target,
        edgeType: e.type,
        fromType,
        toType,
      });
    }
  }

  const allPaths = fileNodes.map((n) => norm(n.filePath || ""));
  const infraFiles = [];
  let hasDockerfile = false;
  let hasCompose = false;
  let hasK8s = false;
  let hasTerraform = false;
  let hasCI = false;
  for (const p of allPaths) {
    const base = basename(p);
    if (/^Dockerfile/i.test(base) || /(^|\/)Dockerfile$/i.test(p)) {
      hasDockerfile = true;
      infraFiles.push(p);
    }
    if (/docker-compose/i.test(base)) {
      hasCompose = true;
      infraFiles.push(p);
    }
    if (/(^|\/)(k8s|kubernetes|helm|charts)\//i.test(p) || /deployment\.ya?ml$/i.test(p)) {
      hasK8s = true;
      infraFiles.push(p);
    }
    if (/\.tf$|\.tfvars$/i.test(p)) {
      hasTerraform = true;
      infraFiles.push(p);
    }
    if (/(^|\/)\.github\/workflows\//i.test(p) || /(^|\/)\.gitlab-ci\.yml$/.test(p) || /(^|\/)Jenkinsfile$/.test(p)) {
      hasCI = true;
      infraFiles.push(p);
    }
    if (base === "Procfile" || base === "Makefile") infraFiles.push(p);
  }

  const schemaFiles = [];
  const migrationFiles = [];
  const dataModelFiles = [];
  const apiHandlerFiles = [];
  for (const n of fileNodes) {
    const p = norm(n.filePath || "");
    const base = basename(p);
    const tags = n.tags || [];
    if (/\.(sql|graphql|gql|prisma)$/i.test(base) || n.type === "schema") schemaFiles.push(p);
    if (/migrations?\//i.test(p)) migrationFiles.push(p);
    if (
      n.type === "table" ||
      tags.includes("data-model") ||
      /(^|\/)(models|entities|db)\//i.test(p) ||
      /models?\.py$|db\.py$/.test(base)
    ) {
      if (n.type === "file" || n.type === "table" || n.type === "schema") dataModelFiles.push(n.type === "file" ? p : n.id);
    }
    if (
      tags.includes("api-handler") ||
      /(^|\/)(routes|routers|controllers|api)\//i.test(p) ||
      /routes\.py$|router\.py$/.test(base)
    ) {
      if (n.type === "file") apiHandlerFiles.push(p);
    }
  }

  const docFiles = fileNodes.filter((n) => n.type === "document" || /\.(md|rst)$/i.test(n.filePath || ""));
  const groupsWithDocsSet = new Set();
  for (const n of docFiles) {
    const g = groupOf(norm(n.filePath || ""), prefix);
    groupsWithDocsSet.add(g);
  }
  const totalGroups = Object.keys(directoryGroups).length;
  const undocumentedGroups = Object.keys(directoryGroups).filter((g) => !groupsWithDocsSet.has(g)).sort();
  const groupsWithDocs = groupsWithDocsSet.size;

  const dependencyDirection = [];
  const seenPairs = new Set();
  for (const row of interGroupImports) {
    const key = [row.from, row.to].sort().join("|");
    if (seenPairs.has(key)) continue;
    seenPairs.add(key);
    const forward = pairCounts[row.from + "\0" + row.to] || 0;
    const backward = pairCounts[row.to + "\0" + row.from] || 0;
    if (forward > backward) dependencyDirection.push({ dependent: row.from, dependsOn: row.to });
    else if (backward > forward) dependencyDirection.push({ dependent: row.to, dependsOn: row.from });
  }

  const pkgDep = [];
  const seenPkg = new Set();
  for (const row of interPackageImports) {
    const key = [row.from, row.to].sort().join("|");
    if (seenPkg.has(key)) continue;
    seenPkg.add(key);
    const forward = pkgPair[row.from + "\0" + row.to] || 0;
    const backward = pkgPair[row.to + "\0" + row.from] || 0;
    if (forward > backward) pkgDep.push({ dependent: row.from, dependsOn: row.to, forward, backward });
    else if (backward > forward) pkgDep.push({ dependent: row.to, dependsOn: row.from, forward: backward, backward: forward });
  }

  const filesPerGroup = {};
  for (const [g, ids] of Object.entries(directoryGroups)) filesPerGroup[g] = ids.length;
  const nodeTypeCounts = {};
  for (const [t, ids] of Object.entries(nodeTypeGroups)) nodeTypeCounts[t] = ids.length;

  const fileFanIn = {};
  const fileFanOut = {};
  for (const n of fileNodes) {
    if ((fanIn[n.id] || 0) > 0) fileFanIn[n.id] = fanIn[n.id];
    if ((fanOut[n.id] || 0) > 0) fileFanOut[n.id] = fanOut[n.id];
  }

  const result = {
    scriptCompleted: true,
    commonPrefix: prefix,
    directoryGroups,
    packageGroups,
    nodeTypeGroups,
    crossCategoryEdges,
    nonCodeLinkCount: nonCodeLinks.length,
    interGroupImports,
    interPackageImports,
    intraGroupDensity,
    patternMatches,
    filePatternMatches,
    deploymentTopology: {
      hasDockerfile,
      hasCompose,
      hasK8s,
      hasTerraform,
      hasCI,
      infraFiles: [...new Set(infraFiles)],
    },
    dataPipeline: {
      schemaFiles: [...new Set(schemaFiles)],
      migrationFiles: [...new Set(migrationFiles)],
      dataModelFiles: [...new Set(dataModelFiles)],
      apiHandlerFiles: [...new Set(apiHandlerFiles)],
    },
    docCoverage: {
      groupsWithDocs,
      totalGroups,
      coverageRatio: totalGroups === 0 ? 0 : Number((groupsWithDocs / totalGroups).toFixed(2)),
      undocumentedGroups,
    },
    dependencyDirection,
    packageDependencyDirection: pkgDep,
    groupImportSets: {
      importsFrom: groupImportsFrom,
      importedBy: groupImportedBy,
    },
    fileStats: {
      totalFileNodes: fileNodes.length,
      filesPerGroup,
      nodeTypeCounts,
    },
    fileFanIn,
    fileFanOut,
    importAdjacency: adjacency,
  };

  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
}

main();
