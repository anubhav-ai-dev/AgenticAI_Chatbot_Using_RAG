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

function depthOf(filePath) {
  if (!filePath) return 99;
  const parts = String(filePath).split(/[\\/]/).filter(Boolean);
  return Math.max(0, parts.length - 1);
}

function isRoot(filePath) {
  return depthOf(filePath) === 0;
}

function isRootOrOneLevel(filePath) {
  const d = depthOf(filePath);
  return d === 0 || d === 1;
}

function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!inputPath || !outputPath) {
    fail("Usage: node ua-tour-analyze.js <input.json> <output.json>");
  }

  let data;
  try {
    data = JSON.parse(fs.readFileSync(inputPath, "utf8"));
  } catch (err) {
    fail("Failed to read input: " + err.message);
  }

  const nodes = Array.isArray(data.nodes) ? data.nodes : [];
  const edges = Array.isArray(data.edges) ? data.edges : [];
  const layers = Array.isArray(data.layers) ? data.layers : [];

  const nodeById = new Map();
  for (const node of nodes) {
    if (node && node.id) nodeById.set(node.id, node);
  }

  const fanInSources = new Map();
  const fanOutTargets = new Map();
  for (const node of nodes) {
    fanInSources.set(node.id, new Set());
    fanOutTargets.set(node.id, new Set());
  }

  for (const edge of edges) {
    if (!edge || !edge.source || !edge.target) continue;
    if (edge.source === edge.target) continue;
    if (fanOutTargets.has(edge.source)) {
      fanOutTargets.get(edge.source).add(edge.target);
    }
    if (fanInSources.has(edge.target)) {
      fanInSources.get(edge.target).add(edge.source);
    }
  }

  const fanInOf = (id) => (fanInSources.get(id) ? fanInSources.get(id).size : 0);
  const fanOutOf = (id) => (fanOutTargets.get(id) ? fanOutTargets.get(id).size : 0);

  const fanInRanking = nodes
    .map((node) => ({
      id: node.id,
      fanIn: fanInOf(node.id),
      name: node.name || node.id,
    }))
    .sort((a, b) => b.fanIn - a.fanIn || a.id.localeCompare(b.id))
    .slice(0, 20);

  const fanOutRanking = nodes
    .map((node) => ({
      id: node.id,
      fanOut: fanOutOf(node.id),
      name: node.name || node.id,
    }))
    .sort((a, b) => b.fanOut - a.fanOut || a.id.localeCompare(b.id))
    .slice(0, 20);

  const n = nodes.length || 1;
  const fanOutValues = nodes.map((node) => fanOutOf(node.id)).sort((a, b) => b - a);
  const fanInValues = nodes.map((node) => fanInOf(node.id)).sort((a, b) => a - b);
  const topFanOutCount = Math.max(1, Math.ceil(n * 0.1));
  const fanOutCutoff = fanOutValues[Math.min(topFanOutCount, fanOutValues.length) - 1] || 0;
  const bottomFanInCount = Math.max(1, Math.ceil(n * 0.25));
  const fanInCutoff = fanInValues[Math.min(bottomFanInCount, fanInValues.length) - 1] || 0;

  const highFanOutIds = new Set(
    nodes.filter((node) => fanOutOf(node.id) >= fanOutCutoff && fanOutOf(node.id) > 0).map((node) => node.id)
  );
  // Keep only the actual top 10% when many share the cutoff.
  const topFanOutIds = new Set(
    [...nodes]
      .sort((a, b) => fanOutOf(b.id) - fanOutOf(a.id) || a.id.localeCompare(b.id))
      .slice(0, topFanOutCount)
      .map((node) => node.id)
  );
  const lowFanInIds = new Set(
    [...nodes]
      .sort((a, b) => fanInOf(a.id) - fanInOf(b.id) || a.id.localeCompare(b.id))
      .slice(0, bottomFanInCount)
      .map((node) => node.id)
  );

  const entryPointCandidates = nodes
    .map((node) => {
      let score = 0;
      const name = node.name || "";
      const filePath = node.filePath || "";
      const type = node.type || "";
      const isDoc = type === "document" || name.toLowerCase().endsWith(".md");
      const isCode = type === "file";

      if (isCode && ENTRY_FILENAMES.has(name)) score += 3;
      if (isCode && isRootOrOneLevel(filePath)) score += 1;
      if (isCode && topFanOutIds.has(node.id)) score += 1;
      if (isCode && lowFanInIds.has(node.id)) score += 1;

      if (isDoc && name.toLowerCase() === "readme.md" && isRoot(filePath)) score += 5;
      else if (isDoc && name.toLowerCase().endsWith(".md") && isRoot(filePath)) score += 2;

      return {
        id: node.id,
        score,
        name: node.name || node.id,
        summary: node.summary || "",
      };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .slice(0, 5);

  const codeEntry = entryPointCandidates.find((item) => {
    const node = nodeById.get(item.id);
    return node && node.type === "file";
  });

  const adjacency = new Map();
  for (const node of nodes) adjacency.set(node.id, new Set());
  for (const edge of edges) {
    if (!edge || (edge.type !== "imports" && edge.type !== "calls")) continue;
    if (!nodeById.has(edge.source) || !nodeById.has(edge.target)) continue;
    if (edge.source === edge.target) continue;
    adjacency.get(edge.source).add(edge.target);
  }

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
      const current = queue.shift();
      bfsTraversal.order.push(current.id);
      bfsTraversal.depthMap[current.id] = current.depth;
      const key = String(current.depth);
      if (!bfsTraversal.byDepth[key]) bfsTraversal.byDepth[key] = [];
      bfsTraversal.byDepth[key].push(current.id);
      const nextIds = [...(adjacency.get(current.id) || [])].sort();
      for (const nextId of nextIds) {
        if (seen.has(nextId)) continue;
        seen.add(nextId);
        queue.push({ id: nextId, depth: current.depth + 1 });
      }
    }
  }

  function pick(node) {
    return {
      id: node.id,
      name: node.name || node.id,
      type: node.type || "",
      summary: node.summary || "",
    };
  }

  const nonCodeFiles = {
    documentation: nodes.filter((node) => node.type === "document").map(pick),
    infrastructure: nodes
      .filter((node) => node.type === "service" || node.type === "pipeline" || node.type === "resource")
      .map(pick),
    data: nodes
      .filter((node) => node.type === "table" || node.type === "schema" || node.type === "endpoint")
      .map(pick),
    config: nodes.filter((node) => node.type === "config").map(pick),
  };

  const pairKey = (a, b) => (a < b ? a + "||" + b : b + "||" + a);
  const importPairs = new Set();
  const callPairs = new Set();
  const undirected = new Map();

  function addUndirected(a, b) {
    if (!a || !b || a === b) return;
    if (!undirected.has(a)) undirected.set(a, new Set());
    if (!undirected.has(b)) undirected.set(b, new Set());
    undirected.get(a).add(b);
    undirected.get(b).add(a);
  }

  const directed = { imports: new Set(), calls: new Set() };
  for (const edge of edges) {
    if (!edge || !edge.source || !edge.target || edge.source === edge.target) continue;
    if (edge.type === "imports" || edge.type === "calls") {
      directed[edge.type].add(edge.source + "->" + edge.target);
    }
    if (nodeById.has(edge.source) && nodeById.has(edge.target)) {
      addUndirected(edge.source, edge.target);
    }
  }

  const bidirectional = [];
  const seenPairs = new Set();
  for (const key of directed.imports) {
    const [source, target] = key.split("->");
    const reverse = target + "->" + source;
    if (!directed.imports.has(reverse)) continue;
    if (!nodeById.has(source) || !nodeById.has(target)) continue;
    const pk = pairKey(source, target);
    if (seenPairs.has(pk)) continue;
    seenPairs.add(pk);
    importPairs.add(pk);
    bidirectional.push([source, target]);
  }
  for (const key of directed.calls) {
    const [source, target] = key.split("->");
    const reverse = target + "->" + source;
    if (!directed.calls.has(reverse)) continue;
    if (!nodeById.has(source) || !nodeById.has(target)) continue;
    const pk = pairKey(source, target);
    if (seenPairs.has(pk)) continue;
    seenPairs.add(pk);
    callPairs.add(pk);
    bidirectional.push([source, target]);
  }

  const parent = new Map();
  function find(id) {
    if (!parent.has(id)) parent.set(id, id);
    if (parent.get(id) !== id) parent.set(id, find(parent.get(id)));
    return parent.get(id);
  }
  function unite(a, b) {
    const pa = find(a);
    const pb = find(b);
    if (pa !== pb) parent.set(pa, pb);
  }
  for (const [a, b] of bidirectional) unite(a, b);

  const groups = new Map();
  for (const [a, b] of bidirectional) {
    const root = find(a);
    if (!groups.has(root)) groups.set(root, new Set());
    groups.get(root).add(a);
    groups.get(root).add(b);
  }

  const edgeListBetween = (members) => {
    const set = new Set(members);
    let count = 0;
    for (const edge of edges) {
      if (!edge) continue;
      if (set.has(edge.source) && set.has(edge.target) && edge.source !== edge.target) count += 1;
    }
    return count;
  };

  const clusters = [];
  for (const members of groups.values()) {
    const cluster = new Set(members);
    let grew = true;
    while (grew && cluster.size < 5) {
      grew = false;
      const candidates = new Map();
      for (const member of cluster) {
        for (const neighbor of undirected.get(member) || []) {
          if (cluster.has(neighbor)) continue;
          candidates.set(neighbor, (candidates.get(neighbor) || 0) + 1);
        }
      }
      const expandable = [...candidates.entries()]
        .filter(([, links]) => links >= 2)
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
      if (!expandable.length) break;
      cluster.add(expandable[0][0]);
      grew = true;
    }
    const nodeIds = [...cluster].slice(0, 5).sort();
    if (nodeIds.length >= 2 && nodeIds.length <= 5) {
      clusters.push({ nodes: nodeIds, edgeCount: edgeListBetween(nodeIds) });
    }
  }

  clusters.sort((a, b) => b.edgeCount - a.edgeCount || b.nodes.length - a.nodes.length || a.nodes[0].localeCompare(b.nodes[0]));
  const topClusters = clusters.slice(0, 10);

  const nodeSummaryIndex = {};
  for (const node of nodes) {
    nodeSummaryIndex[node.id] = {
      name: node.name || node.id,
      type: node.type || "",
      summary: node.summary || "",
    };
  }

  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
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
    scoringNotes: {
      fanOutCutoff,
      fanInCutoff,
      topFanOutCount,
      bottomFanInCount,
      highFanOutConsidered: highFanOutIds.size,
    },
  };

  try {
    fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
  } catch (err) {
    fail("Failed to write output: " + err.message);
  }
}

main();
