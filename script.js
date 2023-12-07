function predictSentiment() {
    var review = document.getElementById('review').value;

    // Send a POST request to the backend
    $.ajax({
        type: 'POST',
        url: '/predict',
        data: { review: review },
        success: function (data) {
            // Display the prediction result and additional information
            $('#result').html('Prediction: ' + (data.prediction === 1 ? 'Positive' : 'Negative') +
                '<br> Sentiment Intensity: ' + data.sentiment_intensity.toFixed(4) +
                '<br> Sentiment Score: ' + data.sentiment_score.toFixed(2) +
                '<br> Word Count: ' + data.word_count +
                '<br> Collocations: ' + JSON.stringify(data.collocations) 
                );
        },
        error: function (error) {
            console.error('Error:', error);
            $('#result').html('Error: ' + error);
        }
    });
}
