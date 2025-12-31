const express = require("express");
const mongoose = require("mongoose");
const multer = require("multer");
const cloudinary = require("../data/cloudinary");
const CaptionQueue = require("../models/CaptionQueue");
require("dotenv").config();


const app = express();
const PORT = process.env.PORT



mongoose.connect(process.env.MONGO_URI);

app.use(express.urlencoded({ extended: true }));

//  memory storage (NO local files)
const upload = multer({ storage: multer.memoryStorage() });

// HTML Form
app.get("/", (req, res) => {
  res.send(`
    <h2>Upload to Cloudinary</h2>
    <form action="/upload" method="POST" enctype="multipart/form-data">
      <input type="file" name="photo" required />
      <br/><br/>
      <input type="text" name="caption" placeholder="Caption" required />
      <br/><br/>
      <button type="submit">Upload</button>
    </form>
  `);
});

// Upload Route (FAST)
app.post("/upload", upload.single("photo"), async (req, res) => {
  try {
    const { caption } = req.body;
    const file = req.file;

    //  Respond immediately 
    res.json({
      message: "Upload received",
      caption
    });

    // Background task
    cloudinary.uploader
      .upload(
        `data:${file.mimetype};base64,${file.buffer.toString("base64")}`,
        { folder: "posts" }
      )
      .then(async (result) => {
        await CaptionQueue.create({
          caption,
          imageId: result.public_id
        });
      })
      .catch((err) => {
        console.error("Background upload failed:", err);
      });

  } catch (err) {
    console.error(err);
    res.status(500).send("Upload failed");
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});