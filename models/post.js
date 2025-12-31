// const mongoose = require("mongoose");

// const postSchema = new mongoose.Schema(
//   {
//     caption: {
//       type: String,
//       required: true
//     },
//     imageId: {
//       type: String, // Cloudinary public_id
//       required: true
//     },    decision: {
//       type: String,
//       enum: ["ABUSIVE", "CLEAN"],
//       required: true
//     },
//     xgboostScore: {
//       type: Number,
//       required: true
//     },
//     finalScore: {
//       type: Number,
//       required: true
//     },
//     matchedWords: {
//       type: [String],
//       default: []
//     },
//     moderatedAt: {
//       type: Date,
//       default: Date.now
//     },
//   },
//   { timestamps: true }
// );

// module.exports = mongoose.model("Post", postSchema);
const mongoose = require("mongoose");

const postSchema = new mongoose.Schema({
  caption: { type: String, required: true },
  imageId: { type: String, required: true },

  decision: {
    type: String,
    enum: ["ABUSIVE", "BORDERLINE", "CLEAN"],
    required: true
  },

  xgboostScore: { type: Number, required: true },
  finalScore: { type: Number, required: true },

  moderatedAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("Post", postSchema);
