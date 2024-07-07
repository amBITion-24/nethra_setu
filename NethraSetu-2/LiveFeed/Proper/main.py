from video_processor import VideoProcessor

if __name__ == "__main__":
    class_thresholds = {
        'bus': 0.3,
        'display': 0.4,
        'dnumber': 0.45
    }

    static_list = ['365J', 'KIA-14', '365P', 'G4', '298MV', '356K', '210P', '266Q', '111C', '215H', '366E', '366Z', '378', '600F', '365']

    anomaly_correction = {
        ']': 'J',
        's': '5',
        'I': '1',
        'l': '1',
    }

    processor = VideoProcessor(
        model_path="/Users/vishnumr/My Files/Programs/Python/Mini Project/Trained Models/best.pt",
        class_thresholds=class_thresholds,
        static_list=static_list,
        anomaly_correction=anomaly_correction
    )

    processor.start_processing()
