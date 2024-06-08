import cv2
import os

def capture_images(name, save_folder, num_images=10):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise Exception("Could not open webcam!")

    count = 1
    while count <= num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image. Exiting.")
            break

        # Display the resulting frame
        cv2.imshow('Capturing Images', frame)

        # Save the image to the specified folder
        image_path = os.path.join(save_folder, f"{name}_{count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image {count} saved as {image_path}")

        count += 1

        # Wait for a short period to simulate manual capture
        cv2.waitKey(500)  # 500 ms delay between captures

        # Press 'q' to quit capturing early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter the name for labeling the images: ")
    save_folder = "images"
    num_images = int(input("Enter the number of images to capture: "))
    capture_images(name, save_folder, num_images)
