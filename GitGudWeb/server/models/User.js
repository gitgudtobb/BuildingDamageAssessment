const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    username: {
        type: String,
        required: [true, 'Username is required'],
        unique: true,
        trim: true,
        minlength: [3, 'Username must be at least 3 characters long']
    },
    email: {
        type: String,
        unique: true,
        trim: true,
        lowercase: true,
        validate: {
            validator: function(v) {
                if (this.authType === 'auth0' && !v) return true;
                return /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/.test(v);
            },
            message: 'Please enter a valid email'
        },
        sparse: true // Boş değerlere izin ver
    },
    name: {
        type: String,
        trim: true
    },
    profilePicture: {
        type: String
    },
    bio: {
        type: String,
        maxlength: [500, 'Bio cannot be more than 500 characters']
    },
    phone: {
        type: String,
        trim: true
    },
    address: {
        type: String,
        trim: true
    },
    auth0Id: {
        type: String,
        unique: true,
        sparse: true
    },
    authType: {
        type: String,
        enum: ['auth0'],
        required: true,
        default: 'auth0'
    },
    roles: {
        type: [String],
        default: ['user']
    },
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    }
});

// Güncelleme zamanını otomatik olarak ayarla
userSchema.pre('save', function(next) {
    this.updatedAt = Date.now();
    next();
});

// Auth0 kullanıcıları için geçici email oluşturma
userSchema.pre('save', function(next) {
    if (this.authType === 'auth0' && !this.email && this.auth0Id) {
        this.email = `temp-${this.auth0Id}@example.com`;
    } else if (this.authType === 'auth0' && this.email) {
        // Email already exists for Auth0 user, no need to generate a temporary one
    }
    next();
});

module.exports = mongoose.model('User', userSchema);
