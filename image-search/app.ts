import {
  AutoModel,
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
} from "@xenova/transformers";
import dotenv from "dotenv";
import { Pinecone, PineconeRecord } from "@pinecone-database/pinecone";
import OpenAI from "openai";
const model = await AutoModel.from_pretrained("Xenova/clip-vit-base-patch16");

dotenv.config();

const pcIndexName = "image-search";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// const response = await openai.chat.completions.create({
//   model: "gpt-3.5-turbo",
//   temperature: 0,
//   messages: [
//     {
//       role: "system",
//       content: `You are an aviation expert. Here is your knowledge to answer the user's question: ${knowledge}`,
//     },
//     { role: "user", content: query },
//   ],
// });

// const pinecone = new Pinecone({
//   apiKey: process.env.PINECONE_API_KEY || "",
// });
async function uploadImage(path: string, isUrl = false) {
  try {
    const processor = await AutoProcessor.from_pretrained(
      "Xenova/clip-vit-base-patch16"
    );
    const visionModel = await CLIPVisionModelWithProjection.from_pretrained(
      "Xenova/clip-vit-base-patch16"
    );

    let img: RawImage;
    if (isUrl) {
      img = await RawImage.read(path);
    } else {
      img = await RawImage.fromURL(path);
    }
    const inputs = await processor(img);

    const { image_embeds } = await visionModel(inputs);
    const vector = Array.from(image_embeds.data) as number[];
    const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
    const index = pc.index(pcIndexName);

    const record: PineconeRecord = {
      id: path,
      values: vector,
      metadata: { file: path },
    };
    await index.upsert([record]);
  } catch (error) {
    console.error(error);
    throw error;
  }
}

async function uploadImages(paths: string[], isUrl = false) {
  try {
    const processor = await AutoProcessor.from_pretrained(
      "Xenova/clip-vit-base-patch16"
    );
    const visionModel = await CLIPVisionModelWithProjection.from_pretrained(
      "Xenova/clip-vit-base-patch16"
    );

    const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
    const index = pc.index(pcIndexName);

    for (const path of paths) {
      const img = isUrl
        ? await RawImage.fromURL(path)
        : await RawImage.read(path);

      const inputs = await processor(img);
      const { image_embeds } = await visionModel(inputs);
      const vector = Array.from(image_embeds.data) as number[];

      const record: PineconeRecord = {
        id: path,
        values: vector,
        metadata: { file: path, name: `Chunk-${paths.indexOf(path)}` },
      };

      await index.upsert([record]);
      console.log(`Chunk ${paths.indexOf(path)} uploaded:`, path);
    }
  } catch (error) {
    console.error(error);
    throw error;
  }
}

async function checkImageExists(path: string, isUrl = false, threshold = 0.8) {
  // Model & Pinecone bağlantısı
  const processor = await AutoProcessor.from_pretrained(
    "Xenova/clip-vit-base-patch16"
  );
  const visionModel = await CLIPVisionModelWithProjection.from_pretrained(
    "Xenova/clip-vit-base-patch16"
  );
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
  const index = pc.index(pcIndexName);

  let img: RawImage;
  if (isUrl) {
    img = await RawImage.read(path);
  } else {
    img = await RawImage.fromURL(path);
  }
  const queryInputs = await processor(img);
  const { image_embeds: qEmbeds } = await visionModel(queryInputs);
  const qVector = Array.from(qEmbeds.data) as number[];

  const queryResponse = await index.query({
    vector: qVector,
    topK: 3,
    includeMetadata: true,
    filter: {
      file: "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/onagho-boyundan-bagli-sari-crop-92d9-4.jpg",
    },
  });

  const matches = queryResponse.matches?.filter(
    (m) => m.score && m.score >= threshold
  );

  if (matches && matches.length > 0) {
    console.log("Eşleşen Resimler", matches);
    return {
      exists: true,
      matches: matches.map((m) => ({
        id: m.id,
        score: m.score,
        metadata: m.metadata,
      })),
    };
  } else {
    console.log("Eşleşen Resim Bulunamadı");
    return { exists: false };
  }
}

// const images = [
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/garias-cepli-uzun-batik-desenli-kahve---b744-.jpg",
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/garias-cepli-uzun-batik-desenli-kahve---acb6-.jpg",
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/garias-cepli-uzun-batik-desenli-bej-et-ff85e9.jpg",
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/garias-cepli-uzun-batik-desenli-bej-et--464a-.jpg",
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/hamper-siyah-file-etek-0-b8bc.jpg",
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/hamper-siyah-file-etek-e1-ad2.jpg",
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/gabbie-boxer-detay-olive-green-mini-et-8ce904.jpg",
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/gabbie-boxer-detay-olive-green-mini-et--39f81.jpg",
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/hanoi-ip-detay-ekru-triko-crop--6afc2.jpg",
//   "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/hanoi-ip-detay-ekru-triko-crop-244053.jpg",
// ];

// await uploadImages(images, true);

await checkImageExists(
  "https://static.ticimax.cloud/cdn-cgi/image/width=1845,quality=99,format=webp/55981/uploads/urunresimleri/buyuk/garias-cepli-uzun-batik-desenli-kahve---acb6-.jpg",
  true,
  0.95
);
