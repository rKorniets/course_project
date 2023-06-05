let filenames = [];

const audioPlayerHtml = '<div><p>"{0}"</p><audio controls><source src="{1}"></audio></div>'

function replase_audio_player(name, path){
    return audioPlayerHtml.replace("{0}", name).replace("{1}", path);
}

function get_plot()
{
    display_names = filenames.length >= 3 ? ['Original', 'Noisy', 'Denoised'] : ['Original', 'Denoised'];
    data = {
        'filenames': filenames,
        'display_names': display_names
    }
    $.ajax({
        url: "/api/audio/plot",
        type: "POST",
        data: JSON.stringify(data),
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function(data)
        {
            console.log(data);
            json = $.parseJSON(JSON.stringify(data));
            $('#plot').append(json['plot']);
        }
    });
}

function get_audio_players(){
    $(".music-player-container").append(replase_audio_player("Original", "/api/audio/download/" + filenames[0]));
    if (filenames.length >= 3){
        $(".music-player-container").append(replase_audio_player("Noisy", "/api/audio/download/" + filenames[1]));
    }
    $(".music-player-container").append(replase_audio_player("Denoised", "/api/audio/download/" + filenames[filenames.length - 1]));
}

function toggle_to_upload()
{
    if (!$('#upload-page').is(':visible')){
        $('#upload-page').fadeIn();
        $('#result-page').fadeOut();
        $('#plot').empty();
        $('.music-player-container').empty();
    }
}

function toggle_to_results() {
    if (!$('#result-page').is(':visible')){
        $('#upload-page').fadeOut();
        $('#result-page').fadeIn();
    }
}

function upload_demo_file()
{
    let form_data = new FormData($("#demo-upload-form")[0]);
    $.ajax({
        url: "/api/demo/upload",
        type: "POST",
        data: form_data,
        contentType: false,
        cache: false,
        processData: false,
        success: function(response)
        {
            console.log(response);
            toggle_to_results();
            var json = $.parseJSON(JSON.stringify(response));
            filenames = json['filenames'];
            get_plot();
            get_audio_players();
        }
    });
}

function upload_real_file()
{
    let form_data = new FormData($("#real-upload-form")[0]);
    $.ajax({
        url: "/api/real/upload",
        type: "POST",
        data: form_data,
        contentType: false,
        cache: false,
        processData: false,
        success: function(response)
        {
            console.log(response);
            toggle_to_results();
            var json = $.parseJSON(JSON.stringify(response));
            filenames = json['filenames'];
            get_plot();
            get_audio_players();
        }
    });
}