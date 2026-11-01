#!/usr/bin/env node
"use strict";

const fs = require("fs");

const ENTRY_FILENAMES = new Set([
  "index.ts", "index.js", "main.ts", "main.js", "app.ts", "app.js",
  "server.ts", "server.js", "mod.rs", "main.go", "main.py", "main.rs",
  "manage.py", "app.py", "wsgi.py", "asgi.py", "run.py", "__main__.py",
  "Application.java", "Main.java", "Program.cs", "config.ru", "index.php",
  "App.swift", "Application.kt", "main.cpp", "main.c",
]);

function fail(message) {
  process.stderr.write(String(message) + "\n");
  process.exit(1);
}

function readJson(path) {
  try {
    return JSON.parse(fs.readFileSync(path, "utf8"));
  } catch (err) {
    fail(`Failed to read ${path}: ${err.message}`);
  }
}

function depthOfPath(filePath) {
  if (!filePath) return 99;
  const parts = String(filePath).split(/[/\\]/).filter(Boolean);
  return parts.length;
}

function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!inputPath || !outputPath) {
    fail("Usage: node ua-tour-analyze.js <input.json> <output.json>");
  }

  const input = readJson(inputPath);
  const nodes = Array.isArray(input.nodes) ? input.nodes : null;
  const edges = Array.isArray(input.edges) ? input.edges : null;
  const layers = Array.isArray(input.layers) ? input.layers : [];
  if (!nodes || !edges) fail("Input must contain nodes and edges arrays");

  const nodeById = new Map();
  for (const node of nodes) {
    if (node && node.id) nodeById.set(node.id, node);
  }

  const fanIn = new Map();
  const fanOut = new Map();
  for (const id of nodeById.keys()) {
    fanIn.set(id, 0);
    fanOut.set(id, 0);
  }

  const adjacency = new Map();
  for (const id of nodeById.keys()) adjacency.set(id, new Set());

  const importFwd = new Map();
  const importRev = new Map();
  const callFwd = new Map();
  const callRev = new Map();
  for (const id of nodeById.keys()) {
    importFwd.set(id, new Set());
    importRev.set(id, new Set());
    callFwd.set(id, new Set());
    callRev.set(id, new Set());
  }

  const undirectedPairs = new Map();
  function pairKey(a, b) {
    return a < b ? `${a}\0${b}` : `${b}\0${a}`;
  }

  let countedEdges = 0;
  for (const edge of edges) {
    if (!edge || !edge.source || !edge.target) continue;
    const sourceKnown = nodeById.has(edge.source);
    const targetKnown = nodeById.has(edge.target);
    if (targetKnown && edge.source !== edge.target) {
      fanIn.set(edge.target, (fanIn.get(edge.target) || 0) + 1);
    }
    if (sourceKnown && edge.source !== edge.target) {
      fanOut.set(edge.source, (fanOut.get(edge.source) || 0) + 1);
    }
    if (!sourceKnown || !targetKnown || edge.source === edge.target) continue;
    countedEdges += 1;
    adjacency.get(edge.source).add(edge.target);
    adjacency.get(edge.target).add(edge.source);
    const key = pairKey(edge.source, edge.target);
    undirectedPairs.set(key, (undirectedPairs.get(key) || 0) + 1);

    if (edge.type === "imports") {
      importFwd.get(edge.source).add(edge.target);
      importRev.get(edge.target).add(edge.source);
    } else if (edge.type === "calls") {
      callFwd.get(edge.source).add(edge.target);
      callRev.get(edge.target).add(edge.source);
    }
  }

  function rank(map, field, limit) {
    return [...map.entries()]
      .map(([id, value]) => ({
        id,
        [field]: value,
        name: nodeById.get(id)?.name || id,
      }))
      .sort((a, b) => b[field] - a[field] || a.id.localeCompare(b.id))
      .slice(0, limit);
  }

  const fanInValues = [...fanIn.values()].sort((a, b) => a - b);
  const fanOutValues = [...fanOut.values()].sort((a, b) => a - b);
  const n = fanInValues.length || 1;
  const bottomCutoff = fanInValues[Math.max(0, Math.floor(n * 0.25) - 1)] ?? 0;
  const topCutoff = fanOutValues[Math.min(n - 1, Math.ceil(n * 0.9))] ?? 0;
  // top 10%: fan-out at or above the value where 90% of nodes are at or below
  const topIndex = Math.max(0, Math.ceil(n * 0.9) - 1);
  const highFanOutThreshold = fanOutValues[Math.min(n - 1, topIndex)] ?? 0;
  const lowIndex = Math.max(0, Math.floor(n * 0.25) - 1);
  const lowFanInThreshold = fanInValues[Math.min(n - 1, lowIndex)] ?? 0;

  const highFanOut = new Set();
  const lowFanIn = new Set();
  for (const [id, value] of fanOut.entries()) {
    if (value >= highFanOutThreshold && value > 0) highFanOut.add(id);
  }
  for (const [id, value] of fanIn.entries()) {
    if (value <= lowFanInThreshold) lowFanIn.add(id);
  }

  const entryPointCandidates = [];
  for (const node of nodes) {
    let score = 0;
    const name = node.name || "";
    const filePath = node.filePath || "";
    const type = node.type || "";
    const parts = depthOfPath(filePath);

    if (type === "file") {
      if (ENTRY_FILENAMES.has(name)) score += 3;
      if (parts <= 2) score += 1;
      if (highFanOut.has(node.id)) score += 1;
      if (lowFanIn.has(node.id)) score += 1;
    } else if (type === "document") {
      const isRoot = parts === 1;
      if (isRoot && name.toLowerCase() === "readme.md") score += 5;
      else if (isRoot && name.toLowerCase().endsWith(".md")) score += 2;
    }

    if (score > 0) {
      entryPointCandidates.push({
        id: node.id,
        score,
        name: node.name,
        summary: node.summary || "",
        type,
      });
    }
  }
  entryPointCandidates.sort(
    (a, b) => b.score - a.score || a.id.localeCompare(b.id)
  );
  const topEntry = entryPointCandidates.slice(0, 5);

  const codeEntry = entryPointCandidates.find(
    (c) => nodeById.get(c.id)?.type === "file"
  );

  const bfsTraversal = {
    startNode: codeEntry ? codeEntry.id : null,
    order: [],
    depthMap: {},
    byDepth: {},
  };

  if (codeEntry) {
    const queue = [{ id: codeEntry.id, depth: 0 }];
    const seen = new Set([codeEntry.id]);
    while (queue.length) {
      const { id, depth } = queue.shift();
      bfsTraversal.order.push(id);
      bfsTraversal.depthMap[id] = depth;
      if (!bfsTraversal.byDepth[depth]) bfsTraversal.byDepth[depth] = [];
      bfsTraversal.byDepth[depth].push(id);
      const next = new Set();
      for (const target of importFwd.get(id) || []) next.add(target);
      for (const target of callFwd.get(id) || []) next.add(target);
      const sorted = [...next].sort((a, b) => a.localeCompare(b));
      for (const target of sorted) {
        if (seen.has(target)) continue;
        seen.add(target);
        queue.push({ id: target, depth: depth + 1 });
      }
    }
  }

  const nonCodeFiles = {
    documentation: [],
    infrastructure: [],
    data: [],
    config: [],
  };
  for (const node of nodes) {
    const item = {
      id: node.id,
      name: node.name,
      type: node.type,
      summary: node.summary || "",
    };
    if (node.type === "document") nonCodeFiles.documentation.push(item);
    else if (node.type === "service" || node.type === "pipeline" || node.type === "resource") {
      nonCodeFiles.infrastructure.push(item);
    } else if (node.type === "table" || node.type === "schema" || node.type === "endpoint") {
      nonCodeFiles.data.push(item);
    } else if (node.type === "config") nonCodeFiles.config.push(item);
  }

  const bidirectional = [];
  const seenPairs = new Set();
  for (const id of nodeById.keys()) {
    for (const other of importFwd.get(id) || []) {
      if (importFwd.get(other)?.has(id)) {
        const key = pairKey(id, other);
        if (!seenPairs.has(key)) {
          seenPairs.add(key);
          bidirectional.push([id, other]);
        }
      }
    }
    for (const other of callFwd.get(id) || []) {
      if (callFwd.get(other)?.has(id)) {
        const key = pairKey(id, other);
        if (!seenPairs.has(key)) {
          seenPairs.add(key);
          bidirectional.push([id, other]);
        }
      }
    }
  }

  const parent = new Map();
  function find(x) {
    if (!parent.has(x)) parent.set(x, x);
    if (parent.get(x) !== x) parent.set(x, find(parent.get(x)));
    return parent.get(x);
  }
  function unite(a, b) {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(rb, ra);
  }
  for (const [a, b] of bidirectional) unite(a, b);

  const groups = new Map();
  for (const id of parent.keys()) {
    const root = find(id);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(id);
  }

  const clusters = [];
  for (const members of groups.values()) {
    let cluster = [...new Set(members)].sort((a, b) => a.localeCompare(b));
    if (cluster.length < 2) continue;
    if (cluster.length > 5) cluster = cluster.slice(0, 5);
    const memberSet = new Set(cluster);
    let expanded = true;
    while (expanded && cluster.length < 5) {
      expanded = false;
      let best = null;
      let bestLinks = 1;
      for (const [id] of nodeById) {
        if (memberSet.has(id)) continue;
        let links = 0;
        for (const member of cluster) {
          if (adjacency.get(id)?.has(member)) links += 1;
        }
        if (links >= 2 && links > bestLinks) {
          best = id;
          bestLinks = links;
        }
      }
      if (best) {
        cluster.push(best);
        memberSet.add(best);
        expanded = true;
      }
    }
    let edgeCount = 0;
    for (let i = 0; i < cluster.length; i++) {
      for (let j = i + 1; j < cluster.length; j++) {
        edgeCount += undirectedPairs.get(pairKey(cluster[i], cluster[j])) || 0;
      }
    }
    clusters.push({
      nodes: cluster.sort((a, b) => a.localeCompare(b)),
      edgeCount,
    });
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount || a.nodes[0].localeCompare(b.nodes[0]));
  const topClusters = clusters.slice(0, 10);

  const nodeSummaryIndex = {};
  for (const node of nodes) {
    nodeSummaryIndex[node.id] = {
      name: node.name,
      type: node.type,
      summary: node.summary || "",
    };
  }

  const result = {
    scriptCompleted: true,
    entryPointCandidates: topEntry,
    fanInRanking: rank(fanIn, "fanIn", 20),
    fanOutRanking: rank(fanOut, "fanOut", 20),
    bfsTraversal,
    nonCodeFiles,
    clusters: topClusters,
    layers: {
      count: layers.length,
      list: layers.map((layer) => ({
        id: layer.id,
        name: layer.name,
        description: layer.description || "",
      })),
    },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length,
    _meta: {
      countedEdgesBetweenKnownNodes: countedEdges,
      highFanOutThreshold,
      lowFanInThreshold,
      bottomCutoff,
      topCutoff,
      bidirectionalPairs: bidirectional.length,
    },
  };

  try {
    fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
  } catch (err) {
    fail(`Failed to write ${outputPath}: ${err.message}`);
  }
}

main();
