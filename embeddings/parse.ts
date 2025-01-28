import { CheerioAPI, load } from "cheerio";
import dotenv from "dotenv";
import fs from "fs";
import { LlamaParseReader } from "llamaindex";
dotenv.config();

const pdfUrl = "./drylab.pdf";

async function fetchPdf() {
  const reader = new LlamaParseReader({ resultType: "markdown" });

  const documents = await reader.loadData(pdfUrl);

  fs.writeFileSync("parsedPdf.json", JSON.stringify(documents, null, 2));
}

async function fetchUrl() {
  const htmlContent = `
  <html>
    <head><title>Test</title></head>
    <body>
      <h1>Merhaba Dünya</h1>
      <p>Bu bir test paragrafıdır.</p>
    </body>
  </html>
`;

  const $: CheerioAPI = load(htmlContent);

  // Parse edilmiş HTML'den veri çıkarma
  const title = $("title").text();
  const heading = $("h1").text();
  const paragraph = $("p").text();

  const parsedContent = `
Başlık: ${title}
Başlık 1: ${heading}
Paragraf: ${paragraph}
`;

  // 2. LlamaIndex ile analiz etme
  const llamaIndex = new LlamaIndex();
  llamaIndex.addDocument(parsedContent);
}

async function main() {
  //   fetchPdf();
  await fetchUrl();
}

main().catch(console.error);
