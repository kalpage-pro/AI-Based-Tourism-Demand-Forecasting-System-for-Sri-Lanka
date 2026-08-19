require('dotenv').config();
const mongoose = require('mongoose');
const TouristDestination = require('../models/TouristDestination.model');

const sampleDestinations = [
  {
    name: 'Sigiriya',
    region: 'Central',
    description: 'Ancient rock fortress with stunning frescoes and landscaped gardens. A UNESCO World Heritage Site.',
    highlights: ['Lion Rock Fortress', 'Ancient Frescoes', 'Water Gardens', 'Mirror Wall'],
    bestTimeToVisit: 'January - April',
    category: 'historical',
    popularity: 95,
    yearlyArrivals: 450000,
    averageStayDays: 2,
    isFeatured: true,
    hotels: [
      {
        name: 'Heritance Kandalama',
        rating: 5,
        priceRange: 'luxury',
        pricePerNight: 250,
        amenities: ['Pool', 'Spa', 'Restaurant', 'WiFi'],
        contact: '+94 66 555 5000'
      },
      {
        name: 'Sigiriya Village Hotel',
        rating: 4,
        priceRange: 'mid-range',
        pricePerNight: 80,
        amenities: ['Pool', 'Restaurant', 'WiFi'],
        contact: '+94 66 228 6803'
      }
    ],
    flights: [
      {
        airline: 'SriLankan Airlines',
        from: 'Colombo',
        price: 150,
        duration: '45 min (helicopter)',
        frequency: 'Daily',
        isEconomical: false
      }
    ],
    coordinates: { latitude: 7.9570, longitude: 80.7603 },
    ratings: { overall: 4.8, totalReviews: 12500 }
  },
  {
    name: 'Kandy',
    region: 'Central',
    description: 'The last royal capital of Sri Lanka, home to the Temple of the Sacred Tooth Relic.',
    highlights: ['Temple of Tooth', 'Kandy Lake', 'Royal Botanical Gardens', 'Cultural Shows'],
    bestTimeToVisit: 'December - April',
    category: 'cultural',
    popularity: 90,
    yearlyArrivals: 520000,
    averageStayDays: 3,
    isFeatured: true,
    hotels: [
      {
        name: 'The Grand Kandyan',
        rating: 5,
        priceRange: 'luxury',
        pricePerNight: 180,
        amenities: ['Pool', 'Spa', 'Restaurant', 'Gym', 'WiFi'],
        contact: '+94 81 223 3888'
      },
      {
        name: 'Hotel Suisse',
        rating: 3,
        priceRange: 'budget',
        pricePerNight: 45,
        amenities: ['Restaurant', 'WiFi'],
        contact: '+94 81 222 2637'
      }
    ],
    flights: [
      {
        airline: 'Bus Service',
        from: 'Colombo',
        price: 8,
        duration: '3.5 hours',
        frequency: 'Every 30 min',
        isEconomical: true
      },
      {
        airline: 'Train',
        from: 'Colombo',
        price: 15,
        duration: '3 hours',
        frequency: '5 times daily',
        isEconomical: true
      }
    ],
    coordinates: { latitude: 7.2906, longitude: 80.6337 },
    ratings: { overall: 4.6, totalReviews: 18200 }
  },
  {
    name: 'Galle Fort',
    region: 'South',
    description: 'Historic Dutch colonial fort, a UNESCO World Heritage Site with charming streets and boutiques.',
    highlights: ['Fort Walls', 'Dutch Reformed Church', 'Lighthouse', 'Art Galleries'],
    bestTimeToVisit: 'November - April',
    category: 'historical',
    popularity: 88,
    yearlyArrivals: 380000,
    averageStayDays: 2,
    isFeatured: true,
    hotels: [
      {
        name: 'Amangalla',
        rating: 5,
        priceRange: 'luxury',
        pricePerNight: 450,
        amenities: ['Pool', 'Spa', 'Fine Dining', 'Butler Service'],
        contact: '+94 91 223 3388'
      },
      {
        name: 'Fort Bazaar',
        rating: 4,
        priceRange: 'mid-range',
        pricePerNight: 120,
        amenities: ['Restaurant', 'WiFi', 'Rooftop'],
        contact: '+94 91 223 2000'
      }
    ],
    flights: [
      {
        airline: 'Express Bus',
        from: 'Colombo',
        price: 6,
        duration: '2.5 hours',
        frequency: 'Every 20 min',
        isEconomical: true
      }
    ],
    coordinates: { latitude: 6.0328, longitude: 80.2170 },
    ratings: { overall: 4.7, totalReviews: 15800 }
  },
  {
    name: 'Yala National Park',
    region: 'South',
    description: 'Sri Lanka\'s most visited wildlife park, famous for having the highest leopard density in the world.',
    highlights: ['Leopard Safari', 'Elephant Herds', 'Bird Watching', 'Camping'],
    bestTimeToVisit: 'February - July',
    category: 'wildlife',
    popularity: 92,
    yearlyArrivals: 280000,
    averageStayDays: 2,
    isFeatured: true,
    hotels: [
      {
        name: 'Cinnamon Wild Yala',
        rating: 4,
        priceRange: 'luxury',
        pricePerNight: 200,
        amenities: ['Safari', 'Pool', 'Restaurant', 'Nature Walks'],
        contact: '+94 47 223 9450'
      },
      {
        name: 'Yala Safari Game Lodge',
        rating: 3,
        priceRange: 'mid-range',
        pricePerNight: 70,
        amenities: ['Safari Tours', 'Restaurant'],
        contact: '+94 47 224 0500'
      }
    ],
    coordinates: { latitude: 6.3699, longitude: 81.5046 },
    ratings: { overall: 4.5, totalReviews: 8900 }
  },
  {
    name: 'Ella',
    region: 'Uva',
    description: 'Picturesque hill country town known for stunning views, hiking trails, and the famous train journey.',
    highlights: ['Nine Arch Bridge', 'Ella Rock', 'Little Adam\'s Peak', 'Tea Plantations'],
    bestTimeToVisit: 'January - March',
    category: 'hill-country',
    popularity: 85,
    yearlyArrivals: 320000,
    averageStayDays: 3,
    isFeatured: true,
    hotels: [
      {
        name: '98 Acres Resort',
        rating: 5,
        priceRange: 'luxury',
        pricePerNight: 280,
        amenities: ['Infinity Pool', 'Spa', 'Tea Lounge', 'Hiking Guides'],
        contact: '+94 57 205 0098'
      },
      {
        name: 'Ella Flower Garden Resort',
        rating: 3,
        priceRange: 'budget',
        pricePerNight: 35,
        amenities: ['Garden', 'Restaurant', 'WiFi'],
        contact: '+94 57 222 8888'
      }
    ],
    flights: [
      {
        airline: 'Scenic Train',
        from: 'Kandy',
        price: 10,
        duration: '6 hours',
        frequency: '3 times daily',
        isEconomical: true
      }
    ],
    coordinates: { latitude: 6.8667, longitude: 81.0466 },
    ratings: { overall: 4.7, totalReviews: 11200 }
  },
  {
    name: 'Mirissa',
    region: 'South',
    description: 'Beautiful coastal town famous for whale watching and pristine beaches.',
    highlights: ['Whale Watching', 'Beach', 'Surfing', 'Secret Beach'],
    bestTimeToVisit: 'November - April',
    category: 'beach',
    popularity: 82,
    yearlyArrivals: 240000,
    averageStayDays: 4,
    isFeatured: false,
    hotels: [
      {
        name: 'Paradise Beach Club',
        rating: 4,
        priceRange: 'mid-range',
        pricePerNight: 90,
        amenities: ['Beach Access', 'Pool', 'Restaurant', 'Water Sports'],
        contact: '+94 41 225 0400'
      },
      {
        name: 'Mirissa Beach Inn',
        rating: 3,
        priceRange: 'budget',
        pricePerNight: 40,
        amenities: ['Beach Access', 'Restaurant', 'WiFi'],
        contact: '+94 41 225 1234'
      }
    ],
    coordinates: { latitude: 5.9480, longitude: 80.4690 },
    ratings: { overall: 4.4, totalReviews: 7500 }
  },
  {
    name: 'Anuradhapura',
    region: 'North Central',
    description: 'Ancient capital of Sri Lanka with sacred Buddhist sites and massive ancient ruins.',
    highlights: ['Sacred Bodhi Tree', 'Ruwanwelisaya', 'Jetavanaramaya', 'Ancient Monasteries'],
    bestTimeToVisit: 'February - September',
    category: 'religious',
    popularity: 78,
    yearlyArrivals: 350000,
    averageStayDays: 2,
    isFeatured: false,
    hotels: [
      {
        name: 'Ulagalla Resort',
        rating: 5,
        priceRange: 'luxury',
        pricePerNight: 320,
        amenities: ['Pool', 'Spa', 'Horse Riding', 'Ayurveda'],
        contact: '+94 25 205 3500'
      },
      {
        name: 'The Sanctuary at Tissawewa',
        rating: 4,
        priceRange: 'mid-range',
        pricePerNight: 85,
        amenities: ['Pool', 'Restaurant', 'Gardens'],
        contact: '+94 25 222 2299'
      }
    ],
    flights: [
      {
        airline: 'Bus Service',
        from: 'Colombo',
        price: 7,
        duration: '4 hours',
        frequency: 'Hourly',
        isEconomical: true
      }
    ],
    coordinates: { latitude: 8.3114, longitude: 80.4037 },
    ratings: { overall: 4.5, totalReviews: 9800 }
  },
  {
    name: 'Nuwara Eliya',
    region: 'Central',
    description: 'Known as "Little England", this hill station offers cool climate, tea estates, and colonial architecture.',
    highlights: ['Tea Estates', 'Gregory Lake', 'Horton Plains', 'Victoria Park'],
    bestTimeToVisit: 'March - May',
    category: 'hill-country',
    popularity: 80,
    yearlyArrivals: 290000,
    averageStayDays: 3,
    isFeatured: false,
    hotels: [
      {
        name: 'Heritance Tea Factory',
        rating: 5,
        priceRange: 'luxury',
        pricePerNight: 220,
        amenities: ['Tea Tours', 'Spa', 'Fine Dining', 'Hiking'],
        contact: '+94 52 555 5000'
      },
      {
        name: 'Grand Hotel',
        rating: 4,
        priceRange: 'mid-range',
        pricePerNight: 95,
        amenities: ['Golf', 'Restaurant', 'Gardens', 'Tennis'],
        contact: '+94 52 222 2881'
      }
    ],
    coordinates: { latitude: 6.9497, longitude: 80.7891 },
    ratings: { overall: 4.3, totalReviews: 8200 }
  },
  {
    name: 'Bentota',
    region: 'South',
    description: 'Popular beach resort town with golden beaches, water sports, and turtle hatcheries.',
    highlights: ['Beach', 'Water Sports', 'Turtle Hatchery', 'Brief Garden'],
    bestTimeToVisit: 'November - April',
    category: 'beach',
    popularity: 75,
    yearlyArrivals: 210000,
    averageStayDays: 4,
    isFeatured: false,
    hotels: [
      {
        name: 'Vivanta by Taj',
        rating: 5,
        priceRange: 'luxury',
        pricePerNight: 195,
        amenities: ['Beach', 'Pool', 'Spa', 'Water Sports'],
        contact: '+94 34 555 5555'
      },
      {
        name: 'Centara Ceysands',
        rating: 4,
        priceRange: 'mid-range',
        pricePerNight: 110,
        amenities: ['Beach', 'Pool', 'Kids Club', 'Restaurant'],
        contact: '+94 34 227 5073'
      }
    ],
    flights: [
      {
        airline: 'Express Train',
        from: 'Colombo',
        price: 5,
        duration: '1.5 hours',
        frequency: 'Hourly',
        isEconomical: true
      }
    ],
    coordinates: { latitude: 6.4213, longitude: 79.9953 },
    ratings: { overall: 4.2, totalReviews: 6800 }
  },
  {
    name: 'Trincomalee',
    region: 'East',
    description: 'Coastal city with natural harbor, pristine beaches, and whale watching opportunities.',
    highlights: ['Nilaveli Beach', 'Pigeon Island', 'Whale Watching', 'Hot Springs'],
    bestTimeToVisit: 'May - September',
    category: 'beach',
    popularity: 72,
    yearlyArrivals: 180000,
    averageStayDays: 4,
    isFeatured: false,
    hotels: [
      {
        name: 'Jungle Beach by Uga',
        rating: 5,
        priceRange: 'luxury',
        pricePerNight: 350,
        amenities: ['Private Beach', 'Pool', 'Spa', 'Diving'],
        contact: '+94 26 567 0000'
      },
      {
        name: 'Pigeon Island Beach Resort',
        rating: 3,
        priceRange: 'budget',
        pricePerNight: 55,
        amenities: ['Beach', 'Restaurant', 'Snorkeling'],
        contact: '+94 26 222 2776'
      }
    ],
    coordinates: { latitude: 8.5874, longitude: 81.2152 },
    ratings: { overall: 4.4, totalReviews: 5200 }
  }
];

async function seedDestinations() {
  try {
    const mongoUri = process.env.MONGODB_URI || process.env.MONGO_URI || 'mongodb://localhost:27017/tourist_prediction';
    
    await mongoose.connect(mongoUri);
    console.log('Connected to MongoDB');

    // Clear existing destinations
    await TouristDestination.deleteMany({});
    console.log('Cleared existing destinations');

    // Insert sample destinations
    const result = await TouristDestination.insertMany(sampleDestinations);
    console.log(`Successfully seeded ${result.length} destinations`);

    // Display summary
    const summary = await TouristDestination.aggregate([
      {
        $group: {
          _id: '$category',
          count: { $sum: 1 },
          totalArrivals: { $sum: '$yearlyArrivals' }
        }
      }
    ]);

    console.log('\nDestination Summary by Category:');
    summary.forEach(cat => {
      console.log(`  ${cat._id}: ${cat.count} destinations, ${cat.totalArrivals.toLocaleString()} yearly arrivals`);
    });

    await mongoose.disconnect();
    console.log('\nDone!');
    process.exit(0);
  } catch (error) {
    console.error('Error seeding destinations:', error);
    process.exit(1);
  }
}

seedDestinations();
