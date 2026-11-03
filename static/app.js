const API_URL = "/api/review";
const MAX_FILE_SIZE = 30 * 1024 * 1024;

const input = document.querySelector("#pdfInput");
const zone = document.querySelector("#dropZone");
const info = document.querySelector("#fileInfo");
const button = document.querySelector("#reviewButton");
const status = document.querySelector("#statusMessage");
const progress = document.querySelector("#progressWrap");
const section = document.querySelector("#reportSection");
const content = document.querySelector("#reportContent");
const subtitle = document.querySelector("#reportSubtitle");

let file = null;
let report = null;

const REVIEWER_NAMES = {
  problem_contribution: "问题与贡献",
  method: "研究方法",
  evaluation: "实验与评价",
  validity: "有效性与局限",
  conclusion: "结论",
};

function createElement(tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text ?? "";

  if (className) {
    element.className = className;
  }

  return element;
}

function setStatus(message, type = "") {
  status.textContent = message;
  status.className = type ? `status ${type}` : "status";
}

function formatScore(score) {
  return score === null || score === undefined
    ? "无法评分"
    : `${score} / 5`;
}

function chooseFile(selectedFile) {
  if (!selectedFile) return;

  const isPdf =
    selectedFile.type === "application/pdf" ||
    selectedFile.name.toLowerCase().endsWith(".pdf");

  if (!isPdf) {
    setStatus("请选择 PDF 文件。", "error");
    return;
  }

  if (selectedFile.size > MAX_FILE_SIZE) {
    setStatus("文件不能超过 30 MB。", "error");
    return;
  }

  file = selectedFile;
  report = null;

  info.replaceChildren(
    createElement("span", selectedFile.name),
    createElement(
      "small",
      `${(selectedFile.size / 1024 / 1024).toFixed(1)} MB`
    )
  );

  info.hidden = false;
  button.disabled = false;
  section.hidden = true;

  setStatus("文件已就绪，可以开始审稿。", "success");
}

function appendListBlock(title, values, className = "") {
  if (!Array.isArray(values) || values.length === 0) return;

  const block = createElement("section", "", `block ${className}`.trim());
  block.append(createElement("h3", title));

  const list = document.createElement("ul");

  values.forEach((value) => {
    const text =
      typeof value === "string"
        ? value
        : JSON.stringify(value, null, 2);

    list.append(createElement("li", text));
  });

  block.append(list);
  content.append(block);
}

function renderReviewDetails(reviews) {
  if (!reviews || typeof reviews !== "object") return;

  const details = createElement("section", "", "block review-details");
  details.append(createElement("h3", "各维度审稿意见"));

  Object.entries(reviews).forEach(([key, review]) => {
    if (!review || typeof review !== "object") return;

    const reviewer = createElement("section", "", "reviewer-card");
    const reviewerName = REVIEWER_NAMES[key] || key.replaceAll("_", " ");

    reviewer.append(
      createElement(
        "h4",
        `${reviewerName} · ${formatScore(review.score)}`
      )
    );

    if (review.score_reason) {
      reviewer.append(
        createElement("p", `评分依据：${review.score_reason}`)
      );
    }

    appendNestedList(reviewer, "优点", review.strengths);
    appendNestedList(reviewer, "问题", review.issues);
    appendNestedList(reviewer, "建议", review.suggestions);
    appendNestedList(
      reviewer,
      "信息不足",
      review.insufficient_information
    );

    details.append(reviewer);
  });

  content.append(details);
}

function appendNestedList(parent, title, values) {
  if (!Array.isArray(values) || values.length === 0) return;

  const wrapper = document.createElement("div");
  wrapper.className = "reviewer-list";

  wrapper.append(createElement("strong", title));

  const list = document.createElement("ul");

  values.forEach((value) => {
    list.append(createElement("li", String(value)));
  });

  wrapper.append(list);
  parent.append(wrapper);
}

function render(reportData) {
  report = reportData;
  content.replaceChildren();

  content.append(
    createElement(
      "h2",
      reportData.title || "论文审稿报告"
    )
  );

  const score = reportData.average_score;
  const scoredCount = reportData.scored_dimension_count;

  const scoreText =
    score === null || score === undefined
      ? "暂无可计算的总分"
      : `平均分：${score} / 5（${scoredCount ?? 0} 个维度参与计算）`;

  content.append(createElement("p", scoreText, "overall-score"));

  if (reportData.summary) {
    const summaryBlock = createElement("section", "", "block");
    summaryBlock.append(createElement("h3", "总体评价"));
    summaryBlock.append(createElement("p", reportData.summary));
    content.append(summaryBlock);
  }

  appendListBlock("论文优点", reportData.strengths);
  appendListBlock("主要问题", reportData.issues);
  appendListBlock("修改建议", reportData.suggestions);
  renderReviewDetails(reportData.reviews);

  section.hidden = false;
  subtitle.textContent = file ? `文件：${file.name}` : "审稿结果";
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

button.addEventListener("click", async () => {
  if (!file) return;

  button.disabled = true;
  progress.hidden = false;
  section.hidden = true;
  setStatus("正在上传并分析论文…", "loading");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    const result = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(
        result.message ||
          result.detail ||
          `请求失败（HTTP ${response.status}）`
      );
    }

    if (result.code !== undefined && result.code !== 200) {
      throw new Error(result.message || "审稿失败");
    }

    const reportData = result.data?.report;

    if (!reportData || typeof reportData !== "object") {
      throw new Error("接口没有返回结构化的审稿报告对象。");
    }

    render(reportData);
    setStatus("审稿完成。", "success");
  } catch (error) {
    setStatus(error.message || "审稿失败，请稍后重试。", "error");
  } finally {
    button.disabled = !file;
    progress.hidden = true;
  }
});

function toMarkdown(reportData) {
  const lines = [
    `# ${reportData.title || "论文审稿报告"}`,
    "",
    `## 总体评分`,
    reportData.average_score === null ||
    reportData.average_score === undefined
      ? "暂无可计算的总分"
      : `${reportData.average_score} / 5（${reportData.scored_dimension_count ?? 0} 个维度参与计算）`,
  ];

  if (reportData.summary) {
    lines.push("", "## 总体评价", reportData.summary);
  }

  appendMarkdownList(lines, "论文优点", reportData.strengths);
  appendMarkdownList(lines, "主要问题", reportData.issues);
  appendMarkdownList(lines, "修改建议", reportData.suggestions);

  if (reportData.reviews && typeof reportData.reviews === "object") {
    lines.push("", "## 各维度审稿意见");

    Object.entries(reportData.reviews).forEach(([key, review]) => {
      if (!review || typeof review !== "object") return;

      const name = REVIEWER_NAMES[key] || key.replaceAll("_", " ");
      lines.push("", `### ${name}`, `评分：${formatScore(review.score)}`);

      if (review.score_reason) {
        lines.push("", `评分依据：${review.score_reason}`);
      }

      appendMarkdownList(lines, "优点", review.strengths);
      appendMarkdownList(lines, "问题", review.issues);
      appendMarkdownList(lines, "建议", review.suggestions);
      appendMarkdownList(
        lines,
        "信息不足",
        review.insufficient_information
      );
    });
  }

  lines.push(
    "",
    "---",
    "*报告由 AI 生成，仅供参考；请结合论文原文判断。*"
  );

  return lines.join("\n");
}

function appendMarkdownList(lines, title, values) {
  if (!Array.isArray(values) || values.length === 0) return;

  lines.push("", `## ${title}`);

  values.forEach((value) => {
    lines.push(`- ${String(value)}`);
  });
}

document.querySelector("#browseButton").addEventListener("click", () => {
  input.click();
});

zone.addEventListener("click", (event) => {
  if (!event.target.closest("button")) {
    input.click();
  }
});

zone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    input.click();
  }
});

input.addEventListener("change", () => {
  chooseFile(input.files?.[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.add("drag");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.remove("drag");
  });
});

zone.addEventListener("drop", (event) => {
  event.preventDefault();
  chooseFile(event.dataTransfer.files?.[0]);
});

document.querySelector("#downloadButton").addEventListener("click", () => {
  if (!report) return;

  const markdown = toMarkdown(report);
  const blob = new Blob(["\uFEFF", markdown], {
    type: "text/markdown;charset=utf-8",
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = "审稿报告.md";
  link.click();

  URL.revokeObjectURL(url);
});




