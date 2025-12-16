import SwiftUI

// MARK: - Content Type Enum

/// Enum representing different types of content that can be displayed
enum ContentType {
    case article(String)
    case image(String)
    case video(URL)
}

// MARK: - Content View Implementations

/// View for displaying article content
struct ArticleView: View {
    let text: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Article")
                .font(.headline)
            Text(text)
                .font(.body)
        }
        .padding()
    }
}

/// View for displaying image content
struct ImageView: View {
    let imageName: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Image")
                .font(.headline)
            Image(imageName)
                .resizable()
                .scaledToFit()
        }
        .padding()
    }
}

/// View for displaying video content
struct VideoView: View {
    let url: URL
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Video")
                .font(.headline)
            Text("Video at: \(url.absoluteString)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
    }
}

// MARK: - SwiftContentViewsFactory

/// Factory for creating content views based on content type
struct SwiftContentViewsFactory {
    
    /// Creates and returns a view for the specified content type
    /// - Parameter type: The type of content to display
    /// - Returns: A type-erased view configured for the content type
    static func makeView(for type: ContentType) -> AnyView {
        switch type {
        case .article(let text):
            return AnyView(ArticleView(text: text))
        case .image(let imageName):
            return AnyView(ImageView(imageName: imageName))
        case .video(let url):
            return AnyView(VideoView(url: url))
        }
    }
}

// MARK: - Usage Example

/// Example view demonstrating how to use the SwiftContentViewsFactory
struct ContentDisplayView: View {
    let contentType: ContentType
    
    var body: some View {
        SwiftContentViewsFactory.makeView(for: contentType)
    }
}

// MARK: - Preview

#if DEBUG
struct ContentDisplayView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            ContentDisplayView(contentType: .article("This is a sample article with some text content."))
                .previewDisplayName("Article")
            
            ContentDisplayView(contentType: .image("placeholder"))
                .previewDisplayName("Image")
            
            if let url = URL(string: "https://example.com/video.mp4") {
                ContentDisplayView(contentType: .video(url))
                    .previewDisplayName("Video")
            }
        }
    }
}
#endif
