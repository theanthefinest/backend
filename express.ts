import express, { Request, Response } from "express";
import axios from "axios";
import cors from "cors";
import { z } from "zod";

const app = express();
const PORT = 4000;

app.use(cors());
app.use(express.json());

// --- Schema for request using Zod to easy debugging ---
const ChatSchema = z.object({
  prompt: z.string().min(1, "Prompt is required"),
  max_new_tokens: z.number().min(1).max(1024).optional(),
  temperature: z.number().min(0).max(1).optional(),
  top_p: z.number().min(0).max(1).optional(),
});

// Proxy endpoint to FastAPI
app.post("/api/chat", async (req: Request, res: Response) => {
  try {
    const parsed = ChatSchema.parse(req.body);

    const fastapiRes = await axios.post("http://127.0.0.1:8000/chat", parsed, {
      timeout: 200_000, 
    });

    res.json(fastapiRes.data);
  } catch (err: any) {
    if (err.name === "ZodError") {
      return res.status(400).json({ error: err.errors });
    }
    console.error("Express error:", err.message);
    res.status(500).json({ error: "Something went wrong" });
  }
});

// Health check
app.get("/api/health", async (_req: Request, res: Response) => {
  try {
    const fastapiRes = await axios.get("http://127.0.0.1:8000/health");
    res.json(fastapiRes.data);
  } catch (err) {
    res.status(500).json({ status: "fastapi-down" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Express Gateway running at http://localhost:${PORT}`);
});
