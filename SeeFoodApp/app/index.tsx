/*
======================================================================
app/index.tsx (Splash Screen)
======================================================================
This is the initial route of the app. It displays a logo for a
short duration and then navigates to the camera screen.
*/
import { useRouter } from 'expo-router';
import React, { useEffect } from 'react';
import { View, Image, StyleSheet } from 'react-native';

export default function SplashScreen() {
  const router = useRouter();

  useEffect(() => {
    // Increased timeout to 4 seconds for a better user experience
    const timer = setTimeout(() => {
      // Use 'replace' to navigate to the camera screen so the user
      // can't go back to the splash screen.
      router.replace('/camera');
    }, 5000);

    // Cleanup function to clear the timer if the component unmounts
    return () => clearTimeout(timer);
  }, [router]);

  return (
    <View style={styles.container}>
      {/* Ensure you have a logo.png in your assets folder */}
      <Image
        source={require('../assets/logo.png')}
        style={styles.logo}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    // Changed background to black
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: {
    width: 150,
    height: 150,
    resizeMode: 'contain',
  },
});
