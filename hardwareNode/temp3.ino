#include <WiFi.h>
#include <HTTPClient.h>
#include <"DHT.h">
#include <PubSubClient.h>
#include <"aqsensor.h">


#define DHTPIN 4     
#define DHTTYPE DHT22   
#define THRESHOLD_1 1000 // Fresh Air threshold

const char* ssid = "REPLACE_WITH_YOUR_SSID";
const char* password = "REPLACE_WITH_YOUR_PASSWORD";

const char* serverName = "http://192.168.1.106:1880/update-sensor";
const char* pubTopic = “publish/…./…..“; // publish/username/apiKeyIn


unsigned long lastTime = 0;

unsigned long timerDelay = 5000;


void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  Serial.println("Connecting");
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
 
  Serial.println("Timer set to 5 seconds (timerDelay variable), it will take 5 seconds before publishing the first reading.");
  Serial.println(F("DHTxx test!"));
  dht.begin();

}

void loop() {
  
  if ((millis() - lastTime) > timerDelay) {
    if(WiFi.status()== WL_CONNECTED){
      float value2 = dht.readHumidity();
    float value1 = dht.readTemperature();
if (isnan(h) || isnan(t) || isnan(f)) {
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }

  float hif = dht.computeHeatIndex(f, h);
  float hic = dht.computeHeatIndex(t, h, false);

  Serial.print(F("Humidity: "));
  Serial.print(value2);
  Serial.print(F("%  Temperature: "));
  Serial.print(value1);
  int value3 = analogRead(A0);
if(value3 < THRESHOLD_1){
Serial.print(Good quality water: “);
} else {
Serial.print(“Poor quality water: “);
}
}
      WiFiClient client;
      HTTPClient http;
    
      http.begin(client, serverName);

      http.addHeader("Content-Type", "application/x-www-form-urlencoded");
      String httpRequestData = "api_key=tPmAT5Ab3j7F9&sensor=BME280&value1=24.25&value2=49.54&value3=1005.14";           
      int httpResponseCode = http.POST(httpRequestData);
      
      http.addHeader("Content-Type", "application/json");
      int httpResponseCode = http.POST("{\"api_key\":\"tPmAT5Ab3j7F9\",\"sensor\":\"BME280\",\"value1\":\"24.25\",\"value2\":\"49.54\",\"value3\":\"1005.14\"}");

      Serial.print("HTTP Response code: ");
      Serial.println(httpResponseCode);
        
      // Free resources
      http.end();
    }
    else {
      Serial.println("WiFi Disconnected");
    }
    lastTime = millis();
  }
}