# Q&A: Spatial design for visionOS - Discover - Apple Developer
Source URL: https://developer.apple.com/news/?id=fi8ne6ji

View in English

## Q&A: Spatial design for visionOS

December 7, 2023

Spatial computing offers unique opportunities and challenges when designing apps and games. At WWDC23, the Apple design team hosted a wide-ranging Q&A to help developers explore designing for visionOS. Here are some highlights from that conversation, including insights on the spectrum of immersion, key moments, and sound design.

### What’s the best way to make a great first impression on this platform?

While it depends on your app, of course, starting in a window is a great way to introduce people to your app and let them control the amount of immersion. We generally recommend not placing people into a fully immersive experience right away — it’s better to make sure they’re oriented in your app before transporting them somewhere else.

### What should I consider when bringing an existing iPadOS or iOS app to visionOS?

Think about a key moment where your app would really shine spatially. For example, in the Photos app for visionOS, opening a panoramic photo makes the image wrap around your field of view. Ask yourself what that potential key moment — an experience that isn’t bound by a screen — is for your app.

From a more tactical perspective, consider how your UI will need to be optimized for visionOS. To learn more, check out “Design for spatial user interfaces”.

#### Design for spatial user interfaces

### Can you say a bit more about what you mean by a “key moment”?

A key moment is a feature or interaction that takes advantage of the unique capabilities of visionOS. (Think of it as a spatial or immersive highlight in your app.) For instance, if you’re creating a writing app, your key moment might be a focus mode in which you immerse someone more fully in an environment or a Spatial Audio soundscape to get them in the creative zone. That’s just not possible on a screen-based device.

### I often use a grid system when designing for iOS and macOS. Does that still apply here?

Definitely! The grid can be very useful for designing windows, and point sizes translate directly between platforms. Things can get more complex when you’re designing elements in 3D, like having nearby controls for a faraway element. To learn more, check out “Principles of spatial design.”

#### Principles of spatial design

### What’s the best way to test Apple Vision Pro experiences without the device?

You can use the visionOS simulator in Xcode to recreate system gestures, like pinch, drag, tap, and zoom.

### What’s the easiest way to make my spatial computing design look polished?

As a starting point, we recommend using the system-provided UI components. Think about hover shapes, how every element appears by default, and how they change when people look directly at them. When building custom components or larger elements like 3D objects, you'll also need to customize your hover effects.

### What interaction or ergonomic design considerations should I keep in mind when designing for visionOS?

Comfort should guide experiences. We recommend keeping your main content in the field of view, so people don't need to move their neck and body too much. The more centered the content is in the field of view, the more comfortable it is for the eyes. It's also important to consider how you use input. Make sure you support system gestures in your app so people have the option to interact with content indirectly (using their eyes to focus an element and hand gestures, like a pinch, to select). For more on design considerations, check out “Design considerations for vision and motion.”

#### Design considerations for vision and motion

### Are there design philosophies for fully immersive experiences? Should the content wrap behind the person’s head, above them, and below them?

Content can be placed anywhere, but we recommend providing only the amount of immersion needed. Apps can create great immersive experiences without taking over people's entire surroundings. To learn more, check out the Human Interface Guidelines.

Human Interface Guidelines: Immersive experiences

### Are there guidelines for creating an environment for a fully immersive experience?

First, your environment should have a ground plane under the feet that aligns with the real world. As you design the specifics of your environment, focus on key details that will create immersion. For example, you don't need to render all the details of a real theater to convey the feeling of being in one. You can also use subtle motion to help bring an environment to life, like the gentle movement of clouds in the Mount Hood environment.

### What else should I consider when designing for spatial computing?

Sound design comes to mind. When designing for other Apple platforms, you may not have placed as much emphasis on creating audio for your interfaces because people often mute sounds on their devices (or it's just not desirable for your current experience). With Apple Vision Pro, sound is crucial to creating a compelling experience.

People are adept at understanding their surroundings through sound, and you can use sound in your visionOS app or game to help people better understand and interact with elements around them. When someone presses a button, for example, an audio cue helps them recognize and confirm their actions. You can position sound spatially in visionOS so that audio comes directly from the item a person interacts with, and the system can use their surroundings to give it the appropriate reverberation and texture. You can even create spatial soundscapes for scenes to make them feel more lifelike and immersive.

For more on designing sound for visionOS, check out “Explore immersive sound design.”

#### Explore immersive sound design

### Learn more

For even more on designing for visionOS, check out more videos, the Human Interface Guidelines, and the Apple Developer website.

#### Develop your first immersive app

#### Get started with building apps for spatial computing

#### Build great games for spatial computing

Human Interface Guidelines

Design for visionOS