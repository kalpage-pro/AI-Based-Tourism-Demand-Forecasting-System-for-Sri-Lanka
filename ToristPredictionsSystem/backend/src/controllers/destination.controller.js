const TouristDestination = require('../models/TouristDestination.model');

// @desc    Get all destinations (Public - users can view)
// @route   GET /api/v1/destinations
// @access  Private
exports.getAllDestinations = async (req, res) => {
  try {
    const { 
      category, 
      region, 
      featured,
      search,
      sortBy = 'popularity',
      limit = 20,
      page = 1 
    } = req.query;

    const query = { isActive: true };

    if (category) query.category = category;
    if (region) query.region = region;
    if (featured === 'true') query.isFeatured = true;
    if (search) {
      query.$text = { $search: search };
    }

    const sortOptions = {};
    if (sortBy === 'popularity') sortOptions.popularity = -1;
    else if (sortBy === 'arrivals') sortOptions.yearlyArrivals = -1;
    else if (sortBy === 'rating') sortOptions['ratings.overall'] = -1;
    else if (sortBy === 'name') sortOptions.name = 1;
    else sortOptions.createdAt = -1;

    const skip = (parseInt(page) - 1) * parseInt(limit);

    const [destinations, total] = await Promise.all([
      TouristDestination.find(query)
        .sort(sortOptions)
        .skip(skip)
        .limit(parseInt(limit))
        .select('-__v'),
      TouristDestination.countDocuments(query)
    ]);

    res.status(200).json({
      success: true,
      count: destinations.length,
      total,
      page: parseInt(page),
      pages: Math.ceil(total / parseInt(limit)),
      data: destinations
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get single destination
// @route   GET /api/v1/destinations/:id
// @access  Private
exports.getDestination = async (req, res) => {
  try {
    const destination = await TouristDestination.findById(req.params.id)
      .populate('addedBy', 'name email');

    if (!destination) {
      return res.status(404).json({
        success: false,
        message: 'Destination not found'
      });
    }

    res.status(200).json({
      success: true,
      data: destination
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Create destination (Admin only)
// @route   POST /api/v1/destinations
// @access  Private/Admin
exports.createDestination = async (req, res) => {
  try {
    req.body.addedBy = req.user.id;

    const destination = await TouristDestination.create(req.body);

    res.status(201).json({
      success: true,
      message: 'Destination created successfully',
      data: destination
    });
  } catch (error) {
    if (error.code === 11000) {
      return res.status(400).json({
        success: false,
        message: 'A destination with this name already exists'
      });
    }
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Update destination (Admin only)
// @route   PUT /api/v1/destinations/:id
// @access  Private/Admin
exports.updateDestination = async (req, res) => {
  try {
    const destination = await TouristDestination.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    );

    if (!destination) {
      return res.status(404).json({
        success: false,
        message: 'Destination not found'
      });
    }

    res.status(200).json({
      success: true,
      message: 'Destination updated successfully',
      data: destination
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Delete destination (Admin only)
// @route   DELETE /api/v1/destinations/:id
// @access  Private/Admin
exports.deleteDestination = async (req, res) => {
  try {
    const destination = await TouristDestination.findByIdAndDelete(req.params.id);

    if (!destination) {
      return res.status(404).json({
        success: false,
        message: 'Destination not found'
      });
    }

    res.status(200).json({
      success: true,
      message: 'Destination deleted successfully'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get featured destinations
// @route   GET /api/v1/destinations/featured
// @access  Private
exports.getFeaturedDestinations = async (req, res) => {
  try {
    const destinations = await TouristDestination.find({ 
      isFeatured: true, 
      isActive: true 
    })
      .sort({ popularity: -1 })
      .limit(6)
      .select('name region category description mainImage yearlyArrivals ratings popularity');

    res.status(200).json({
      success: true,
      count: destinations.length,
      data: destinations
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get destinations statistics for analytics
// @route   GET /api/v1/destinations/analytics
// @access  Private
exports.getDestinationAnalytics = async (req, res) => {
  try {
    // Get arrivals by category
    const arrivalsByCategory = await TouristDestination.aggregate([
      { $match: { isActive: true } },
      {
        $group: {
          _id: '$category',
          totalArrivals: { $sum: '$yearlyArrivals' },
          avgPopularity: { $avg: '$popularity' },
          count: { $sum: 1 }
        }
      },
      { $sort: { totalArrivals: -1 } }
    ]);

    // Get arrivals by region
    const arrivalsByRegion = await TouristDestination.aggregate([
      { $match: { isActive: true } },
      {
        $group: {
          _id: '$region',
          totalArrivals: { $sum: '$yearlyArrivals' },
          avgPopularity: { $avg: '$popularity' },
          count: { $sum: 1 }
        }
      },
      { $sort: { totalArrivals: -1 } }
    ]);

    // Get top destinations
    const topDestinations = await TouristDestination.find({ isActive: true })
      .sort({ yearlyArrivals: -1 })
      .limit(10)
      .select('name region category yearlyArrivals popularity ratings');

    // Get overall stats
    const overallStats = await TouristDestination.aggregate([
      { $match: { isActive: true } },
      {
        $group: {
          _id: null,
          totalDestinations: { $sum: 1 },
          totalArrivals: { $sum: '$yearlyArrivals' },
          avgPopularity: { $avg: '$popularity' },
          avgRating: { $avg: '$ratings.overall' },
          avgStayDays: { $avg: '$averageStayDays' }
        }
      }
    ]);

    // Category distribution for pie chart
    const categoryDistribution = arrivalsByCategory.map(cat => ({
      name: cat._id ? cat._id.charAt(0).toUpperCase() + cat._id.slice(1).replace('-', ' ') : 'Unknown',
      value: cat.totalArrivals,
      count: cat.count
    }));

    // Region distribution for pie chart
    const regionDistribution = arrivalsByRegion.map(reg => ({
      name: reg._id || 'Unknown',
      value: reg.totalArrivals,
      count: reg.count
    }));

    res.status(200).json({
      success: true,
      data: {
        overview: overallStats[0] || {
          totalDestinations: 0,
          totalArrivals: 0,
          avgPopularity: 0,
          avgRating: 0,
          avgStayDays: 0
        },
        arrivalsByCategory,
        arrivalsByRegion,
        categoryDistribution,
        regionDistribution,
        topDestinations
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Add hotel to destination (Admin only)
// @route   POST /api/v1/destinations/:id/hotels
// @access  Private/Admin
exports.addHotel = async (req, res) => {
  try {
    const destination = await TouristDestination.findById(req.params.id);

    if (!destination) {
      return res.status(404).json({
        success: false,
        message: 'Destination not found'
      });
    }

    destination.hotels.push(req.body);
    await destination.save();

    res.status(201).json({
      success: true,
      message: 'Hotel added successfully',
      data: destination
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Update hotel in destination (Admin only)
// @route   PUT /api/v1/destinations/:id/hotels/:hotelId
// @access  Private/Admin
exports.updateHotel = async (req, res) => {
  try {
    const destination = await TouristDestination.findById(req.params.id);

    if (!destination) {
      return res.status(404).json({
        success: false,
        message: 'Destination not found'
      });
    }

    const hotel = destination.hotels.id(req.params.hotelId);
    if (!hotel) {
      return res.status(404).json({
        success: false,
        message: 'Hotel not found'
      });
    }

    Object.assign(hotel, req.body);
    await destination.save();

    res.status(200).json({
      success: true,
      message: 'Hotel updated successfully',
      data: destination
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Delete hotel from destination (Admin only)
// @route   DELETE /api/v1/destinations/:id/hotels/:hotelId
// @access  Private/Admin
exports.deleteHotel = async (req, res) => {
  try {
    const destination = await TouristDestination.findById(req.params.id);

    if (!destination) {
      return res.status(404).json({
        success: false,
        message: 'Destination not found'
      });
    }

    destination.hotels.pull(req.params.hotelId);
    await destination.save();

    res.status(200).json({
      success: true,
      message: 'Hotel deleted successfully',
      data: destination
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Add flight to destination (Admin only)
// @route   POST /api/v1/destinations/:id/flights
// @access  Private/Admin
exports.addFlight = async (req, res) => {
  try {
    const destination = await TouristDestination.findById(req.params.id);

    if (!destination) {
      return res.status(404).json({
        success: false,
        message: 'Destination not found'
      });
    }

    destination.flights.push(req.body);
    await destination.save();

    res.status(201).json({
      success: true,
      message: 'Flight added successfully',
      data: destination
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Delete flight from destination (Admin only)
// @route   DELETE /api/v1/destinations/:id/flights/:flightId
// @access  Private/Admin
exports.deleteFlight = async (req, res) => {
  try {
    const destination = await TouristDestination.findById(req.params.id);

    if (!destination) {
      return res.status(404).json({
        success: false,
        message: 'Destination not found'
      });
    }

    destination.flights.pull(req.params.flightId);
    await destination.save();

    res.status(200).json({
      success: true,
      message: 'Flight deleted successfully',
      data: destination
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Toggle featured status (Admin only)
// @route   PATCH /api/v1/destinations/:id/toggle-featured
// @access  Private/Admin
exports.toggleFeatured = async (req, res) => {
  try {
    const destination = await TouristDestination.findById(req.params.id);

    if (!destination) {
      return res.status(404).json({
        success: false,
        message: 'Destination not found'
      });
    }

    destination.isFeatured = !destination.isFeatured;
    await destination.save();

    res.status(200).json({
      success: true,
      message: `Destination ${destination.isFeatured ? 'marked as featured' : 'removed from featured'}`,
      data: destination
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get economical flights
// @route   GET /api/v1/destinations/economical-flights
// @access  Private
exports.getEconomicalFlights = async (req, res) => {
  try {
    const destinations = await TouristDestination.find({
      isActive: true,
      'flights.isEconomical': true
    })
      .select('name region flights')
      .lean();

    const economicalFlights = [];
    destinations.forEach(dest => {
      dest.flights.filter(f => f.isEconomical).forEach(flight => {
        economicalFlights.push({
          destination: dest.name,
          region: dest.region,
          ...flight
        });
      });
    });

    // Sort by price
    economicalFlights.sort((a, b) => (a.price || 0) - (b.price || 0));

    res.status(200).json({
      success: true,
      count: economicalFlights.length,
      data: economicalFlights
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get best hotels
// @route   GET /api/v1/destinations/best-hotels
// @access  Private
exports.getBestHotels = async (req, res) => {
  try {
    const { priceRange, minRating } = req.query;

    const destinations = await TouristDestination.find({
      isActive: true,
      hotels: { $exists: true, $ne: [] }
    })
      .select('name region hotels')
      .lean();

    let hotels = [];
    destinations.forEach(dest => {
      dest.hotels.forEach(hotel => {
        let include = true;
        if (priceRange && hotel.priceRange !== priceRange) include = false;
        if (minRating && hotel.rating < parseFloat(minRating)) include = false;
        
        if (include) {
          hotels.push({
            destination: dest.name,
            region: dest.region,
            ...hotel
          });
        }
      });
    });

    // Sort by rating
    hotels.sort((a, b) => (b.rating || 0) - (a.rating || 0));

    res.status(200).json({
      success: true,
      count: hotels.length,
      data: hotels.slice(0, 50)
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};
