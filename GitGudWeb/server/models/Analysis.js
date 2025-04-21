const mongoose = require('mongoose');

const analysisSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  location: {
    type: {
      type: String,
      enum: ['Point'],
      default: 'Point'
    },
    coordinates: {
      type: [Number],
      required: true
    },
    address: String
  },
  images: {
    before: String,
    after: String
  },
  results: {
    damageLevel: {
      type: String,
      enum: ['minor', 'moderate', 'severe', 'critical'],
    },
    damagePercentage: {
      type: Number
    },
    affectedAreas: [String],
    recommendations: [String],
    riskAssessment: String,
    estimatedRepairCost: {
      min: Number,
      max: Number,
      currency: String
    }
  },
  metadata: {
    buildingType: String,
    constructionYear: Number,
    floorCount: Number
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  status: {
    type: String,
    enum: ['pending', 'completed', 'failed'],
    default: 'pending'
  }
});

// Lokasyon bazlı sorgular için indeks
analysisSchema.index({ location: '2dsphere' });

// Kullanıcı bazlı sorgular için indeks
analysisSchema.index({ user: 1, createdAt: -1 });

const Analysis = mongoose.model('Analysis', analysisSchema);

module.exports = Analysis;
