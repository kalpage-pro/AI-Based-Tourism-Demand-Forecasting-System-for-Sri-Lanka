const PDFDocument = require('pdfkit');
const { Parser } = require('json2csv');
const Prediction = require('../models/Prediction.model');
const HistoricalData = require('../models/HistoricalData.model');
const ActivityLog = require('../models/ActivityLog.model');

// Export predictions to CSV
exports.exportPredictionsCSV = async (req, res, next) => {
  try {
    const { startDate, endDate, predictionType } = req.query;
    
    const query = { user: req.user._id };
    
    if (startDate || endDate) {
      query.createdAt = {};
      if (startDate) query.createdAt.$gte = new Date(startDate);
      if (endDate) query.createdAt.$lte = new Date(endDate);
    }
    
    if (predictionType && predictionType !== 'all') {
      query.predictionType = predictionType;
    }

    const predictions = await Prediction.find(query).sort({ createdAt: -1 });
    
    const data = predictions.map(p => ({
      date: p.createdAt.toISOString().split('T')[0],
      year: p.inputData.year,
      month: p.inputData.month,
      predictionType: p.predictionType,
      touristArrivals: p.predictions.touristArrivals?.value || '',
      revenue: p.predictions.revenue?.value || '',
      occupancyRate: p.predictions.rooms?.value || '',
      dollarRate: p.inputData.dollarRate || '',
      confidence: p.predictions.touristArrivals?.confidence || ''
    }));

    const parser = new Parser();
    const csv = parser.parse(data);

    // Log activity
    await ActivityLog.log(req.user._id, 'EXPORT_GENERATED', { format: 'csv', count: data.length }, req);

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename=predictions_${Date.now()}.csv`);
    res.status(200).send(csv);
  } catch (error) {
    next(error);
  }
};

// Export predictions to PDF report
exports.exportPredictionsPDF = async (req, res, next) => {
  try {
    const { startDate, endDate, includeCharts = false } = req.query;
    
    const query = { user: req.user._id };
    if (startDate || endDate) {
      query.createdAt = {};
      if (startDate) query.createdAt.$gte = new Date(startDate);
      if (endDate) query.createdAt.$lte = new Date(endDate);
    }

    const predictions = await Prediction.find(query).sort({ createdAt: -1 }).limit(50);
    
    // Create PDF document
    const doc = new PDFDocument({ margin: 50 });
    
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename=tourism_report_${Date.now()}.pdf`);
    
    doc.pipe(res);

    // Title
    doc.fontSize(24).font('Helvetica-Bold')
       .text('Sri Lanka Tourism Prediction Report', { align: 'center' });
    doc.moveDown();
    
    // Report info
    doc.fontSize(12).font('Helvetica')
       .text(`Generated: ${new Date().toLocaleDateString()}`, { align: 'center' })
       .text(`Total Predictions: ${predictions.length}`, { align: 'center' });
    doc.moveDown(2);

    // Summary statistics
    doc.fontSize(16).font('Helvetica-Bold').text('Summary Statistics');
    doc.moveDown(0.5);
    
    const arrivals = predictions.filter(p => p.predictions.touristArrivals?.value)
                                .map(p => p.predictions.touristArrivals.value);
    const revenues = predictions.filter(p => p.predictions.revenue?.value)
                               .map(p => p.predictions.revenue.value);
    
    if (arrivals.length > 0) {
      const avgArrivals = Math.round(arrivals.reduce((a, b) => a + b, 0) / arrivals.length);
      const maxArrivals = Math.round(Math.max(...arrivals));
      const minArrivals = Math.round(Math.min(...arrivals));
      
      doc.fontSize(11).font('Helvetica')
         .text(`Average Predicted Arrivals: ${avgArrivals.toLocaleString()}`)
         .text(`Maximum Predicted Arrivals: ${maxArrivals.toLocaleString()}`)
         .text(`Minimum Predicted Arrivals: ${minArrivals.toLocaleString()}`);
    }
    
    if (revenues.length > 0) {
      const avgRevenue = Math.round(revenues.reduce((a, b) => a + b, 0) / revenues.length);
      doc.text(`Average Predicted Revenue: $${avgRevenue.toLocaleString()}`);
    }
    
    doc.moveDown(2);

    // Predictions table
    doc.fontSize(16).font('Helvetica-Bold').text('Recent Predictions');
    doc.moveDown(0.5);
    
    // Table header
    const tableTop = doc.y;
    const col1 = 50, col2 = 120, col3 = 180, col4 = 280, col5 = 380;
    
    doc.fontSize(10).font('Helvetica-Bold')
       .text('Date', col1, tableTop)
       .text('Period', col2, tableTop)
       .text('Type', col3, tableTop)
       .text('Arrivals', col4, tableTop)
       .text('Revenue', col5, tableTop);
    
    doc.moveTo(50, tableTop + 15).lineTo(550, tableTop + 15).stroke();
    
    let yPosition = tableTop + 25;
    
    predictions.slice(0, 20).forEach((pred, index) => {
      if (yPosition > 700) {
        doc.addPage();
        yPosition = 50;
      }
      
      doc.fontSize(9).font('Helvetica')
         .text(pred.createdAt.toLocaleDateString(), col1, yPosition)
         .text(`${pred.inputData.month}/${pred.inputData.year}`, col2, yPosition)
         .text(pred.predictionType, col3, yPosition)
         .text(pred.predictions.touristArrivals?.value?.toLocaleString() || '-', col4, yPosition)
         .text(pred.predictions.revenue?.value ? `$${Math.round(pred.predictions.revenue.value).toLocaleString()}` : '-', col5, yPosition);
      
      yPosition += 20;
    });

    // Footer
    doc.fontSize(8).font('Helvetica')
       .text('Generated by Sri Lanka Tourism Prediction System', 50, 750, { align: 'center' });

    doc.end();

    // Log activity
    await ActivityLog.log(req.user._id, 'EXPORT_GENERATED', { format: 'pdf', count: predictions.length }, req);
  } catch (error) {
    next(error);
  }
};

// Export historical data
exports.exportHistoricalData = async (req, res, next) => {
  try {
    const { format = 'csv', startYear, endYear } = req.query;
    
    const query = {};
    if (startYear) query.year = { $gte: parseInt(startYear) };
    if (endYear) query.year = { ...query.year, $lte: parseInt(endYear) };

    const data = await HistoricalData.find(query).sort({ year: 1, month: 1 });

    if (format === 'csv') {
      const csvData = data.map(d => ({
        year: d.year,
        month: d.month,
        touristArrivals: d.touristArrivals,
        revenue: d.revenue,
        rooms: d.rooms,
        dollarRate: d.dollarRate,
        avgStayDuration: d.avgStayDuration
      }));

      const parser = new Parser();
      const csv = parser.parse(csvData);

      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename=historical_data_${Date.now()}.csv`);
      res.status(200).send(csv);
    } else {
      res.status(200).json({
        success: true,
        data
      });
    }
  } catch (error) {
    next(error);
  }
};

// Get export history
exports.getExportHistory = async (req, res, next) => {
  try {
    const exports = await ActivityLog.find({
      user: req.user._id,
      action: 'EXPORT_GENERATED'
    }).sort({ createdAt: -1 }).limit(20);

    res.status(200).json({
      success: true,
      data: exports
    });
  } catch (error) {
    next(error);
  }
};
