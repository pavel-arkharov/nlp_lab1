"use strict";

const numberFormat = new Intl.NumberFormat("en-US");
const charts = new Map();
const chartRenderers = new Map();
const palette = ["#2d6cdf", "#e05a47", "#21867a", "#d99a2b", "#7a5aa6", "#4d7c8a"];
let labData = null;

function formatNumber(value, decimals = null) {
  if (decimals !== null) {
    return Number(value).toFixed(decimals);
  }
  return numberFormat.format(value);
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function replaceDefinitionList(id, items) {
  const list = document.getElementById(id);
  list.replaceChildren();
  items.forEach(([label, value]) => {
    const wrapper = document.createElement("div");
    const term = document.createElement("dt");
    const definition = document.createElement("dd");
    term.textContent = label;
    definition.textContent = value;
    wrapper.append(term, definition);
    list.append(wrapper);
  });
}

function renderTable(id, captionText, columns, rows) {
  const container = document.getElementById(id);
  const table = document.createElement("table");
  const caption = document.createElement("caption");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  const body = document.createElement("tbody");

  caption.textContent = captionText;
  columns.forEach((column) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = column;
    headRow.append(cell);
  });
  rows.forEach((row) => {
    const tableRow = document.createElement("tr");
    row.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      tableRow.append(cell);
    });
    body.append(tableRow);
  });
  head.append(headRow);
  table.append(caption, head, body);
  container.replaceChildren(table);
}

function axisStyle(name) {
  return {
    name,
    nameLocation: "middle",
    nameGap: 38,
    nameTextStyle: { color: "#64706a", fontSize: 12 },
    axisLine: { lineStyle: { color: "#aab5b0" } },
    axisTick: { lineStyle: { color: "#aab5b0" } },
    axisLabel: { color: "#48534e", fontSize: 11 },
    splitLine: { lineStyle: { color: "#e1e6e3" } }
  };
}

function baseOption(title) {
  return {
    animationDuration: 850,
    animationEasing: "cubicOut",
    backgroundColor: "transparent",
    color: palette,
    textStyle: {
      color: "#1f2925",
      fontFamily: '"Avenir Next", Avenir, "Segoe UI", Helvetica, Arial, sans-serif'
    },
    title: {
      text: title,
      left: 8,
      top: 14,
      textStyle: { fontSize: 16, fontWeight: 700, color: "#1f2925" }
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#1f2925",
      borderWidth: 0,
      textStyle: { color: "#ffffff" }
    },
    aria: { enabled: true }
  };
}

function registerChart(elementId, renderer) {
  chartRenderers.set(elementId, renderer);
}

function renderChart(elementId) {
  const container = document.getElementById(elementId);
  if (!container || container.offsetParent === null || !window.echarts) {
    return;
  }
  let chart = charts.get(elementId);
  if (!chart) {
    chart = window.echarts.init(container, null, { renderer: "canvas" });
    charts.set(elementId, chart);
  }
  chart.setOption(chartRenderers.get(elementId)(), true);
  chart.resize();
}

function renderChartsForTask(taskElement) {
  taskElement.querySelectorAll(".chart").forEach((chartElement) => {
    if (chartRenderers.has(chartElement.id)) {
      renderChart(chartElement.id);
    }
  });
}

function horizontalBarOption(title, records, color) {
  const ordered = [...records].reverse();
  return {
    ...baseOption(title),
    grid: { left: 78, right: 40, top: 64, bottom: 50 },
    xAxis: { type: "value", ...axisStyle("Frequency") },
    yAxis: {
      type: "category",
      data: ordered.map((record) => record.word),
      ...axisStyle("Word"),
      axisLabel: { color: "#48534e", fontSize: 11, interval: 0 }
    },
    series: [{
      type: "bar",
      data: ordered.map((record) => record.count),
      itemStyle: { color, borderRadius: [0, 3, 3, 0] },
      label: { show: true, position: "right", color: "#1f2925", fontSize: 10 },
      animationDelay: (index) => index * 22
    }]
  };
}

function registerCharts(data) {
  registerChart("chart-top-words", () =>
    horizontalBarOption("Twenty most frequent words", data.task_2.top_20, palette[0])
  );

  registerChart("chart-rank-frequency", () => ({
    ...baseOption("Rank-frequency distribution"),
    grid: { left: 68, right: 24, top: 66, bottom: 60 },
    tooltip: {
      ...baseOption("").tooltip,
      formatter: (items) => {
        const value = items[0].value;
        return `Rank ${formatNumber(value[0])}<br>Frequency ${formatNumber(value[1])}`;
      }
    },
    xAxis: { type: "log", logBase: 10, ...axisStyle("Word rank (log scale)") },
    yAxis: { type: "log", logBase: 10, ...axisStyle("Frequency (log scale)") },
    series: [{
      name: "Frequency",
      type: "line",
      showSymbol: false,
      smooth: 0.12,
      lineStyle: { width: 3, color: palette[0] },
      areaStyle: { color: "rgba(45, 108, 223, 0.10)" },
      data: data.task_3.rank_frequency.map((record) => [record.rank, record.frequency])
    }]
  }));

  registerChart("chart-coverage", () => ({
    ...baseOption("Cumulative token coverage"),
    grid: { left: 66, right: 24, top: 66, bottom: 60 },
    xAxis: { type: "log", logBase: 10, ...axisStyle("Word types included (log scale)") },
    yAxis: { type: "value", min: 0, max: 100, ...axisStyle("Coverage (%)") },
    series: [{
      name: "Coverage",
      type: "line",
      showSymbol: false,
      smooth: true,
      lineStyle: { width: 3, color: palette[1] },
      areaStyle: { color: "rgba(224, 90, 71, 0.10)" },
      data: data.task_3.rank_frequency.map((record) => [record.rank, record.coverage_percent])
    }]
  }));

  registerChart("chart-least-words", () =>
    horizontalBarOption("Thirty least-frequent words", data.task_4.least_30, palette[1])
  );
  registerChart("chart-middle-words", () =>
    horizontalBarOption("Thirty middle-frequency words", data.task_4.middle_30, palette[3])
  );

  registerChart("chart-word-lengths", () => ({
    ...baseOption("Word length versus frequency"),
    legend: { top: 18, right: 8, data: ["Token frequency", "Distinct types"] },
    grid: { left: 70, right: 72, top: 76, bottom: 58 },
    xAxis: {
      type: "category",
      data: data.task_5.word_lengths.map((record) => record.length),
      ...axisStyle("Word length (characters)")
    },
    yAxis: [
      { type: "value", ...axisStyle("Token frequency") },
      { type: "value", ...axisStyle("Distinct word types"), splitLine: { show: false } }
    ],
    series: [
      {
        name: "Token frequency",
        type: "bar",
        data: data.task_5.word_lengths.map((record) => record.token_count),
        itemStyle: { color: palette[2], borderRadius: [3, 3, 0, 0] }
      },
      {
        name: "Distinct types",
        type: "line",
        yAxisIndex: 1,
        data: data.task_5.word_lengths.map((record) => record.type_count),
        lineStyle: { width: 2.5, color: palette[3] },
        itemStyle: { color: palette[3] }
      }
    ]
  }));

  registerChart("chart-modals", () => ({
    ...baseOption("Modal-word frequencies"),
    grid: { left: 68, right: 24, top: 68, bottom: 54 },
    xAxis: {
      type: "category",
      data: data.task_6.modals.map((record) => record.word),
      ...axisStyle("Modal word")
    },
    yAxis: { type: "value", ...axisStyle("Occurrences") },
    series: [{
      type: "bar",
      data: data.task_6.modals.map((record, index) => ({
        value: record.occurrences,
        itemStyle: { color: palette[index], borderRadius: [4, 4, 0, 0] }
      })),
      label: { show: true, position: "top", color: "#1f2925", fontWeight: 700 }
    }]
  }));

  function modalBoxOption(title, key, axisLabel) {
    return {
      ...baseOption(title),
      grid: { left: 66, right: 24, top: 68, bottom: 52 },
      tooltip: {
        ...baseOption("").tooltip,
        formatter: (item) => {
          const values = item.data.value;
          return `${item.name}<br>Min: ${values[0]}<br>Q1: ${values[1]}<br>Median: ${values[2]}<br>Q3: ${values[3]}<br>Max: ${values[4]}`;
        }
      },
      xAxis: {
        type: "category",
        data: data.task_6.modals.map((record) => record.word),
        ...axisStyle("Modal word")
      },
      yAxis: { type: "value", ...axisStyle(axisLabel) },
      series: [{
        type: "boxplot",
        data: data.task_6.modals.map((record, index) => {
          const summary = record[key];
          return {
            value: [summary.minimum, summary.q1, summary.median, summary.q3, summary.maximum],
            itemStyle: { color: palette[index], borderColor: "#1f2925" }
          };
        })
      }]
    };
  }

  registerChart("chart-modal-words", () =>
    modalBoxOption("Matching post lengths", "word_length", "Post length (words)")
  );
  registerChart("chart-modal-characters", () =>
    modalBoxOption("Matching character lengths", "character_length", "Post length (characters)")
  );

  function densityOption(title, records, axisLabel) {
    const maximum = Math.log10(Math.max(...records.map((record) => record.sentences)));
    return {
      ...baseOption(title),
      grid: { left: 66, right: 58, top: 72, bottom: 58 },
      tooltip: {
        trigger: "item",
        backgroundColor: "#1f2925",
        borderWidth: 0,
        textStyle: { color: "#ffffff" },
        formatter: (item) => {
          const [length, stops, sentences] = item.value;
          return `${axisLabel}: ${formatNumber(length)}<br>Stopwords: ${formatNumber(stops)}<br>Sentences: ${formatNumber(sentences)}`;
        }
      },
      visualMap: {
        min: 0,
        max: maximum,
        dimension: 3,
        orient: "vertical",
        right: 0,
        top: 76,
        text: ["More", "Less"],
        calculable: false,
        inRange: { color: ["#d8e5e1", "#21867a", "#d99a2b", "#e05a47"] },
        textStyle: { color: "#64706a", fontSize: 10 }
      },
      xAxis: { type: "value", ...axisStyle(axisLabel) },
      yAxis: { type: "value", ...axisStyle("Stopword count") },
      series: [{
        type: "scatter",
        progressive: 5000,
        data: records.map((record) => [record.length, record.stopwords, record.sentences, Math.log10(record.sentences)]),
        symbolSize: (value) => Math.min(30, 4 + value[3] * 5.5),
        itemStyle: { opacity: 0.78, borderColor: "rgba(31,41,37,0.22)", borderWidth: 0.5 }
      }]
    };
  }

  registerChart("chart-brown-words", () =>
    densityOption("Stopwords versus word length", data.task_7.word_density, "Sentence length (words)")
  );
  registerChart("chart-brown-characters", () =>
    densityOption(
      `Stopwords vs. characters (${data.task_7.character_bin_width}-char bins)`,
      data.task_7.character_density,
      "Sentence length (characters)"
    )
  );
}

function populateAnswers(data) {
  const task1 = data.task_1;
  const task7 = data.task_7;

  replaceDefinitionList("headline-metrics", [
    ["NPS posts", formatNumber(task1.posts)],
    ["Word tokens", formatNumber(task1.word_tokens)],
    ["Vocabulary", formatNumber(task1.vocabulary_size)],
    ["Brown sentences", formatNumber(task7.sentences)]
  ]);

  setText("result-1", `${formatNumber(task1.posts)} posts`);
  setText("result-2", `${data.task_2.top_20[0].word}: ${formatNumber(data.task_2.top_20[0].count)}`);
  setText("result-3", `${formatNumber(data.task_3.full_vocabulary_rows)} word types`);
  setText("result-4", `target ${formatNumber(data.task_4.middle_target, 2)}`);
  const peakLength = data.task_5.word_lengths.reduce((best, item) => item.token_count > best.token_count ? item : best);
  setText("result-5", `peak: ${peakLength.length} characters`);
  const modalTotal = data.task_6.modals.reduce((total, modal) => total + modal.occurrences, 0);
  setText("result-6", `${formatNumber(modalTotal)} occurrences`);
  setText("result-7", `r = ${formatNumber(task7.correlation_words, 3)}`);

  setText(
    "task-1-answer",
    `NPS Chat contains ${formatNumber(task1.posts)} posts and ${formatNumber(task1.raw_tokens)} raw tokens. After normalization, ${formatNumber(task1.word_tokens)} word tokens remain, representing ${formatNumber(task1.vocabulary_size)} distinct word types.`
  );
  setText("task-1-method", task1.normalization);
  replaceDefinitionList("task-1-stats", [
    ["Posts", formatNumber(task1.posts)],
    ["Raw tokens", formatNumber(task1.raw_tokens)],
    ["Word tokens", formatNumber(task1.word_tokens)],
    ["Vocabulary", formatNumber(task1.vocabulary_size)],
    ["Lexical diversity", formatNumber(task1.lexical_diversity, 4)]
  ]);

  renderTable(
    "table-top-words",
    "Twenty most frequent words",
    ["Rank", "Word", "Frequency"],
    data.task_2.top_20.map((record, index) => [index + 1, record.word, formatNumber(record.count)])
  );

  setText("least-method", data.task_4.least_method);
  setText("middle-method", `${data.task_4.middle_method} The resulting target is ${formatNumber(data.task_4.middle_target, 2)} occurrences.`);
  renderTable(
    "table-least-words",
    "Thirty least-frequent words",
    ["Word", "Frequency"],
    data.task_4.least_30.map((record) => [record.word, formatNumber(record.count)])
  );
  renderTable(
    "table-middle-words",
    "Thirty middle-frequency words",
    ["Word", "Frequency"],
    data.task_4.middle_30.map((record) => [record.word, formatNumber(record.count)])
  );

  renderTable(
    "table-word-lengths",
    "Frequency grouped by word length",
    ["Characters", "Token frequency", "Distinct types"],
    data.task_5.word_lengths.map((record) => [
      record.length,
      formatNumber(record.token_count),
      formatNumber(record.type_count)
    ])
  );

  setText(
    "modal-method",
    `${data.task_6.sentence_unit} Every modal occurrence is counted; each matching post contributes once to that modal's length statistics.`
  );
  renderTable(
    "table-modals",
    "Modal frequency and matching-post length statistics",
    ["Modal", "Occurrences", "Posts", "Mean words", "Median words", "Mean chars", "Median chars"],
    data.task_6.modals.map((record) => [
      record.word,
      formatNumber(record.occurrences),
      formatNumber(record.posts),
      formatNumber(record.word_length.mean, 2),
      formatNumber(record.word_length.median, 2),
      formatNumber(record.character_length.mean, 2),
      formatNumber(record.character_length.median, 2)
    ])
  );

  setText(
    "brown-answer",
    `Across ${formatNumber(task7.sentences)} Brown sentences, NLTK's ${formatNumber(task7.stopword_list_size)}-word English stopword list produces ${formatNumber(task7.total_stopword_occurrences)} matches. Longer sentences strongly tend to contain more stopwords.`
  );
  replaceDefinitionList("task-7-stats", [
    ["Sentences", formatNumber(task7.sentences)],
    ["Stopword list", formatNumber(task7.stopword_list_size)],
    ["Occurrences", formatNumber(task7.total_stopword_occurrences)],
    ["Mean / sentence", formatNumber(task7.mean_stopwords_per_sentence, 2)],
    ["Word-length r", formatNumber(task7.correlation_words, 4)]
  ]);

  setText("generated-at", `Generated ${data.meta.generated_at} with NLTK ${data.meta.nltk_version}`);
}

function initializeControls() {
  const details = [...document.querySelectorAll("details.task")];
  const select = document.getElementById("section-select");

  select.addEventListener("change", () => {
    const selected = document.getElementById(select.value);
    selected.open = true;
    window.requestAnimationFrame(() => {
      renderChartsForTask(selected);
      selected.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  document.getElementById("expand-all").addEventListener("click", () => {
    details.forEach((detail) => {
      detail.open = true;
      window.requestAnimationFrame(() => renderChartsForTask(detail));
    });
  });

  document.getElementById("collapse-all").addEventListener("click", () => {
    details.forEach((detail) => {
      detail.open = false;
    });
  });

  details.forEach((detail) => {
    detail.addEventListener("toggle", () => {
      if (detail.open && labData) {
        window.requestAnimationFrame(() => renderChartsForTask(detail));
      }
    });
  });

  let resizeFrame = null;
  window.addEventListener("resize", () => {
    if (resizeFrame) {
      window.cancelAnimationFrame(resizeFrame);
    }
    resizeFrame = window.requestAnimationFrame(() => {
      charts.forEach((chart) => chart.resize());
    });
  });
}

function initializeWordField() {
  const canvas = document.getElementById("word-field");
  const context = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const vocabulary = ["corpus", "token", "frequency", "word", "length", "chat", "modal", "sentence", "brown", "nltk", "can", "might", "will"];
  let width = 0;
  let height = 0;
  let particles = [];

  function resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    particles = Array.from({ length: Math.max(15, Math.floor(width / 70)) }, (_, index) => ({
      word: vocabulary[index % vocabulary.length],
      x: Math.random() * width,
      y: Math.random() * height,
      speed: 0.08 + Math.random() * 0.18,
      opacity: 0.035 + Math.random() * 0.04,
      size: 10 + Math.random() * 5
    }));
  }

  function draw() {
    context.clearRect(0, 0, width, height);
    context.textBaseline = "middle";
    particles.forEach((particle) => {
      context.font = `600 ${particle.size}px ui-monospace, SFMono-Regular, Consolas, monospace`;
      context.fillStyle = `rgba(31, 41, 37, ${particle.opacity})`;
      context.fillText(particle.word, particle.x, particle.y);
      if (!reduceMotion) {
        particle.y -= particle.speed;
        if (particle.y < -20) {
          particle.y = height + 20;
          particle.x = Math.random() * width;
        }
      }
    });
    if (!reduceMotion) {
      window.requestAnimationFrame(draw);
    }
  }

  resize();
  draw();
  window.addEventListener("resize", resize);
}

async function initializeReport() {
  const loadState = document.getElementById("load-state");
  const configuration = window.LAB_CONFIG || {};
  setText("student-name", configuration.studentName || "Pavel Arkharov");
  setText("course-name", configuration.courseName || "NLP / Natural Language Processing");
  initializeControls();
  initializeWordField();

  try {
    const response = await fetch("data/lab_results.json");
    if (!response.ok) {
      throw new Error(`Data request failed with status ${response.status}`);
    }
    labData = await response.json();
    populateAnswers(labData);
    registerCharts(labData);
    document.querySelectorAll("details.task[open]").forEach((detail) => renderChartsForTask(detail));
    loadState.hidden = true;
  } catch (error) {
    loadState.classList.add("error");
    loadState.textContent = "The generated analysis data could not be loaded. Run the Python analysis, then serve the docs folder through a local web server.";
    console.error(error);
  }
}

document.addEventListener("DOMContentLoaded", initializeReport);
