from langchain_text_splitters import HTMLHeaderTextSplitter
doc="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sample Page</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
    header, footer { background: #f4f4f4; text-align: center; padding: 10px; }
    nav ul { list-style: none; display: flex; justify-content: center; gap: 15px; padding: 0; }
    nav a { text-decoration: none; color: #333; }
    main { padding: 20px; background: #fafafa; }
  </style>
</head>
<body>
  <header><h1>Welcome</h1></header>
  <header><h2>Online business</h2></header>
  <nav>
    <ul>
      <li><a href="#">Home</a></li>
      <li><a href="#">About</a></li>
      <li><a href="#">Contact</a></li>
    </ul>
  </nav>
  <main>
    <p>This is a simple 25-line HTML page with header, nav, content, and footer.</p>
  </main>
  <footer><p>© 2025 My Website</p></footer>
</body>
</html>
"""

header_to_split = [("h1","Header 1"),("h2","Header 2")]

htmlsplitter= HTMLHeaderTextSplitter(header_to_split)
htmlsplitter_output= htmlsplitter.split_text(doc)
print(htmlsplitter_output)
