from docutils import nodes
from sphinx.transforms import SphinxTransform


class ReplaceVideoLinksTransform(SphinxTransform):
    default_priority = 900

    def apply(self):
        video_link = "https://github.com/user-attachments/assets/"

        for node in self.document.traverse(nodes.reference):
            uri = node.get("refuri", "")
            if uri.startswith(video_link):
                video_html = f"""
<video controls width="100%" height="auto" playsinline>
<source src="{uri}" type="video/mp4">
Your browser does not support the video tag.
</video>
"""

                raw_node = nodes.raw("", video_html, format="html")
                parent = node.parent
                index = parent.index(node)
                parent.replace(node, raw_node)
