import dotenv from "dotenv";
dotenv.config();

import pdf from "@cyber2024/pdf-parse-fixed";
import fs from "fs";
import {
  Document,
  TextNode,
  VectorStoreIndex,
  serviceContextFromDefaults,
  storageContextFromDefaults,
} from "llamaindex";
import OpenAI from "openai";

const buffer = fs.readFileSync("./thesis.pdf");
const parsedPdf = await pdf(buffer);

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

console.log("Creating index");
const index = await VectorStoreIndex.fromDocuments([document], {
  serviceContext,
  storageContext,
});
console.log("Index created");

const query = "Which technologies are used to solve congestion at airports?";
const retriever = index.asRetriever();
const matchingNodes = await retriever.retrieve(query);

console.log("Matching nodes", matchingNodes);

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

console.log("response", response.choices[0]);
