$(document).ready(function() {
    $('#company').change(function() {
        var company = $(this).val();
        $.getJSON('/get_models/' + company, function(data) {
            var options = '<option value="">Select a model</option>';
            $.each(data, function(key, value) {
                options += '<option value="' + value + '">' + value + '</option>';
            });
            $('#car_models').html(options);
        });
    });

    $('#prediction-form').submit(function(e) {
        e.preventDefault();
        $('.loading').show();
        $('.error-message').hide();
        $('#prediction').text('');
        $.ajax({
            url: '/predict',
            type: 'post',
            data: $('#prediction-form').serialize(),
            success: function(response) {
                if (response.error) {
                    $('.error-message').text(response.error).show();
                } else {
                    $('#prediction').text('Predicted Price: ' + response.prediction);
                }
            },
            error: function(xhr, status, error) {
                $('.error-message').text('An error occurred: ' + error).show();
            },
            complete: function() {
                $('.loading').hide();
            }
        });
    });
});