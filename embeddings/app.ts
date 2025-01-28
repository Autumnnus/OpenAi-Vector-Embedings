import dotenv from "dotenv";

import pdf from "@cyber2024/pdf-parse-fixed";
import fs from "fs";
import {
  Document,
  LlamaParseReader,
  TextNode,
  VectorStoreIndex,
  serviceContextFromDefaults,
  storageContextFromDefaults,
} from "llamaindex";
import OpenAI from "openai";

dotenv.config();
const pdfUrl = "./drylab.pdf";

const buffer = fs.readFileSync(pdfUrl);
const parsedPdf = await pdf(buffer);

const reader = new LlamaParseReader({ resultType: "markdown" });

// Load and parse the document
// const documents = await reader.loadData("https://www.qpien.com/");
const documents = await reader.loadData(pdfUrl);

console.log("documents", documents);
// console.log("parsedPdf", parsedPdf);
main().catch((error) => console.error("Hata:", error));
// const pdfParser = new PDFParser(this, 1);
// const filename = "./drylab.pdf";
// pdfParser.on("pdfParser_dataError", (errData) =>
//   console.error(errData.parserError)
// );
// pdfParser.on("pdfParser_dataReady", (pdfData) => {
//   console.log({ textContent: pdfParser.getRawTextContent() });
// });
// pdfParser.loadPDF(filename);

const serviceContext = serviceContextFromDefaults({
  chunkSize: 4000,
  chunkOverlap: 500,
});
const storageContext = await storageContextFromDefaults({
  persistDir: "./storage",
});

const document = new Document({
  text: parsedPdf.text,
});
// console.log("document", document);
// console.log("Creating index");
const index = await VectorStoreIndex.fromDocuments([document], {
  serviceContext,
  storageContext,
});
// console.log("Index created", index.asRetriever());

const query = "Which technologies are used to solve congestion at airports?";
const retriever = index.asRetriever();
const matchingNodes = await retriever.retrieve(query);

// console.log("Matching nodes", matchingNodes);

const knowledge = matchingNodes
  ?.map((node) => {
    const textNode = node.node as TextNode;
    return textNode.text;
  })
  .join("\n\n");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const response = await openai.chat.completions.create({
  model: "gpt-3.5-turbo",
  temperature: 0,
  messages: [
    {
      role: "system",
      content: `You are an aviation expert. Here is your knowledge to answer the user's question: ${knowledge}`,
    },
    { role: "user", content: query },
  ],
});

// console.log("response", response.choices[0]);

async function extractParagraphs(pdfPath: string): Promise<string[]> {
  const pdfBuffer = fs.readFileSync(pdfPath);

  // PDF'den metni çıkar
  const data = await pdf(pdfBuffer);

  // Satırları analiz et
  const lines = data.text.split("\n"); // PDF'den satır satır metin çıkar
  const paragraphs: string[] = [];
  let currentParagraph = "";

  for (const line of lines) {
    const trimmedLine = line.trim();

    // Eğer satır boşsa, paragraf tamamlandı demektir
    if (!trimmedLine) {
      if (currentParagraph) {
        paragraphs.push(currentParagraph.trim());
        currentParagraph = ""; // Yeni paragrafa geç
      }
    } else {
      // Satır boş değilse, aynı paragrafta devam et
      currentParagraph += (currentParagraph ? " " : "") + trimmedLine;
    }
  }

  // Kalan metni paragraf olarak ekle
  if (currentParagraph) {
    paragraphs.push(currentParagraph.trim());
  }

  return paragraphs;
}

async function main() {
  const paragraphs = await extractParagraphs(pdfUrl);

  // paragraphs.forEach((para, index) => {
  //   console.log(`\n[Paragraf ${index + 1}]: ${para}`);
  // });
}
