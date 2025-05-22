document.addEventListener('DOMContentLoaded', function() {
    // Make captions clickable and add toggle functionality
    var captions = document.getElementsByClassName('caption-text');
    for (var i = 0; i < captions.length; i++) {
        var caption = captions[i];
        caption.style.cursor = 'pointer';
        caption.onclick = function() {
            var ul = this.parentElement.nextElementSibling;
            if (ul) {
                if (ul.style.display === 'none') {
                    ul.style.display = 'block';
                } else {
                    ul.style.display = 'none';
                }
            }
        };
    }
});
