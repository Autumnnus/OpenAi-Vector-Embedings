import dotenv from "dotenv";
dotenv.config();

import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const response = await openai.chat.completions.create({
  model: "gpt-3.5-turbo",
  temperature: 0,
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "What is the meaning of life?" },
  ],
});

console.log("response", response);
 