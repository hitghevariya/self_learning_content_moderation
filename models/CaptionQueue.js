const mongoose = require("mongoose");

const captionQueueSchema = new mongoose.Schema(
  {
    caption: {
      type: String,
      required: true
    },
    imageId: {
      type: String, // Cloudinary public_id
      required: true
    },
    status: {
      type: String,
      enum: ["pending", "processing"],
      default: "pending"
    }
  },
  { timestamps: true }
);

module.exports = mongoose.model("CaptionQueue", captionQueueSchema);
