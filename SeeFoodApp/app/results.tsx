
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { FoodPrediction } from '../services/api';

export default function ResultsScreen() {
  const { predictionData } = useLocalSearchParams<{ predictionData: string }>();
  const router = useRouter();
  
  let prediction: FoodPrediction | null = null;
  if (predictionData) {
    try {
      prediction = JSON.parse(predictionData);
    } catch (e) {
      console.error("Failed to parse prediction data:", e);
    }
  }

  const handleTryAgain = () => {
    // Use replace to go back to the camera for a clean navigation stack
    router.replace('/camera');
  };

  if (!prediction) {
    return (
      <View style={styles.container}>
        <Text style={styles.resultText}>Could not identify food.</Text>
        <TouchableOpacity style={styles.button} onPress={handleTryAgain}>
          <Text style={styles.buttonText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.resultText}>
          {prediction.name}
        </Text>
        <Text style={styles.confidenceText}>
          Per 100 gm
        </Text>

        <View style={styles.nutritionContainer}>
          <Text style={styles.nutritionTitle}>Nutritional Information</Text>
          <View style={styles.nutritionTable}>
            {Object.entries(prediction.nutrition).map(([key, value]) => (
              <View key={key} style={styles.nutritionRow}>
                <Text style={styles.nutritionLabel}>{key}</Text>
                <Text style={styles.nutritionValue}>{value}</Text>
              </View>
            ))}
          </View>
        </View>
      </View>
      
      <TouchableOpacity style={styles.button} onPress={handleTryAgain}>
        <Text style={styles.buttonText}>Try Again</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1c1c1e',
    justifyContent: 'space-between', // Pushes content and button apart
    padding: 20,
    paddingBottom: 40, // Add padding at the bottom
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  resultText: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#34C759',
    textTransform: 'capitalize',
    textAlign: 'center',
  },
  confidenceText: {
    fontSize: 18,
    color: '#8e8e93',
    marginTop: 10,
    marginBottom: 40,
  },
  nutritionContainer: {
    width: '100%',
    maxWidth: 400,
    backgroundColor: '#2c2c2e',
    borderRadius: 12,
    padding: 20,
  },
  nutritionTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
    marginBottom: 20,
  },
  nutritionTable: {
    width: '100%',
  },
  nutritionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#444',
  },
  nutritionLabel: {
    fontSize: 16,
    color: '#e5e5e7',
  },
  nutritionValue: {
    fontSize: 16,
    color: 'white',
    fontWeight: '600',
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 12,
    alignSelf: 'stretch', // Make button stretch to container padding
    marginTop: 20,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
