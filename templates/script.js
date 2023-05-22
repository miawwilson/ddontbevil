function generateTagCloud(query) {
    $.getJSON('/new_tag_cloud', {'query': query}, function(data) {
        if (data.error) {
            alert(data.error);
        } else {
            var tag_data = data.tag_data;
            var image_bytes = data.image_bytes;
            // do something with tag_data and image_bytes, e.g. update the tag cloud and image in the page
        }
    });
}
