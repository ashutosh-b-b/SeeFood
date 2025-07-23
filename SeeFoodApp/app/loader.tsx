/*
======================================================================
app/loader.tsx (Dedicated Loader Screen)
======================================================================
This new screen receives an image URI, shows a loader, performs
the base64 conversion and API call, then navigates to the results.
*/
import React, { useEffect } from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import * as FileSystem from 'expo-file-system';
import { predictFood } from '../services/api';
import Loader from '../components/Loader';

export default function LoadingScreen() {
  const router = useRouter();
  const { uri } = useLocalSearchParams<{ uri: string }>();

  useEffect(() => {
    if (!uri) {
      Alert.alert("Error", "No image was provided.");
      router.back();
      return;
    }

    const processImage = async () => {
      try {
        // Step 1: Convert the image file to a base64 string.
        // This is the heavy operation we moved from the camera screen.
        const base64 = await FileSystem.readAsStringAsync(uri, {
          encoding: FileSystem.EncodingType.Base64,
        });

        // Step 2: Call the API with the base64 data.
        const prediction = await predictFood(base64);

        // Step 3: Navigate to the results page, passing the full prediction
        // object as a stringified JSON.
        router.replace({
          pathname: '/results',
          params: { predictionData: JSON.stringify(prediction) },
        });

      } catch (error) {
        console.error("Failed to process image:", error);
        Alert.alert("Error", "Could not process the image. Please try again.");
        router.back(); // Go back to the camera on error
      }
    };

    processImage();
  }, [uri]);

  return (
    <View style={styles.container}>
      <Loader />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1c1c1e',
  },
});
