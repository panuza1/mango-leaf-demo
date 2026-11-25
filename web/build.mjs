import { mkdir, copyFile, writeFile } from "node:fs/promises";
const backend = process.env.MANGO_BACKEND_URL || "";
if (backend && new URL(backend).protocol !== "https:") {
  throw new Error("MANGO_BACKEND_URL must use HTTPS");
}
await mkdir("dist", { recursive: true });
await copyFile("index.html", "dist/index.html");
await writeFile("dist/config.js", "window.MANGO_BACKEND_URL = " + JSON.stringify(backend).replace(/</g, "\\u003c") + ";\n");
