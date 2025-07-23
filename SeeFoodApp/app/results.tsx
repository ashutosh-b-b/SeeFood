/*
======================================================================
app/results.tsx (Results Screen)
======================================================================
This screen is now a simple display component. It receives the
final prediction data from the loader screen and renders it.
*/
import React from 'react';
import { View, Text, StyleSheet, Alert } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { FoodPrediction } from '../services/api';

export default function ResultsScreen() {
  const { predictionData } = useLocalSearchParams<{ predictionData: string }>();
  
  let prediction: FoodPrediction | null = null;
  if (predictionData) {
    try {
      prediction = JSON.parse(predictionData);
    } catch (e) {
      console.error("Failed to parse prediction data:", e);
    }
  }

  if (!prediction) {
    return (
      <View style={styles.container}>
        <Text style={styles.resultText}>Could not identify food.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.resultText}>
        {prediction.name}
      </Text>
      <Text style={styles.confidenceText}>
        Confidence: {(prediction.confidence * 100).toFixed(0)}%
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
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1c1c1e',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
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
});
