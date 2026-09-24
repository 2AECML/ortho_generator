#include <E57SimpleReader.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#endif

namespace
{
   constexpr char kMagic[8] = {'O', 'R', 'T', 'H', '5', '7', 'V', '1'};

#pragma pack(push, 1)
   struct Header
   {
      char magic[8];
      std::uint32_t scanCount;
      std::uint32_t flags;
      std::uint64_t pointCount;
   };
#pragma pack(pop)

   template <typename T> void writeValue( const T &value )
   {
      std::cout.write( reinterpret_cast<const char *>( &value ), sizeof( T ) );
      if ( !std::cout )
         throw std::runtime_error( "Failed to write E57 reader output." );
   }

   template <typename T> void writeArray( const std::vector<T> &values, std::size_t count )
   {
      std::cout.write( reinterpret_cast<const char *>( values.data() ),
                       static_cast<std::streamsize>( count * sizeof( T ) ) );
      if ( !std::cout )
         throw std::runtime_error( "Failed to write E57 reader array." );
   }

#ifdef _WIN32
   std::string utf8( const wchar_t *value )
   {
      const int length = WideCharToMultiByte( CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr );
      if ( length <= 1 )
         return {};
      std::string result( static_cast<std::size_t>( length ), '\0' );
      WideCharToMultiByte( CP_UTF8, 0, value, -1, result.data(), length, nullptr, nullptr );
      result.pop_back();
      return result;
   }
#endif

   bool hasCartesian( const e57::Data3D &header )
   {
      return header.pointFields.cartesianXField && header.pointFields.cartesianYField &&
             header.pointFields.cartesianZField;
   }

   bool hasSpherical( const e57::Data3D &header )
   {
      return header.pointFields.sphericalRangeField && header.pointFields.sphericalAzimuthField &&
             header.pointFields.sphericalElevationField;
   }

   bool hasColour( const e57::Data3D &header )
   {
      return header.pointFields.colorRedField && header.pointFields.colorGreenField &&
             header.pointFields.colorBlueField;
   }

   void applyPose( const e57::RigidBodyTransform &pose, double &x, double &y, double &z )
   {
      double qw = pose.rotation.w;
      double qx = pose.rotation.x;
      double qy = pose.rotation.y;
      double qz = pose.rotation.z;
      const double norm = std::sqrt( qw * qw + qx * qx + qy * qy + qz * qz );
      if ( norm > 0.0 )
      {
         qw /= norm;
         qx /= norm;
         qy /= norm;
         qz /= norm;
      }
      const double tx = 2.0 * ( qy * z - qz * y );
      const double ty = 2.0 * ( qz * x - qx * z );
      const double tz = 2.0 * ( qx * y - qy * x );
      const double rx = x + qw * tx + ( qy * tz - qz * ty );
      const double ry = y + qw * ty + ( qz * tx - qx * tz );
      const double rz = z + qw * tz + ( qx * ty - qy * tx );
      x = rx + pose.translation.x;
      y = ry + pose.translation.y;
      z = rz + pose.translation.z;
   }

   int run( const std::string &path, std::size_t capacity )
   {
      if ( capacity == 0 )
         throw std::runtime_error( "Buffer capacity must be positive." );
#ifdef _WIN32
      if ( _setmode( _fileno( stdout ), _O_BINARY ) == -1 )
         throw std::runtime_error( "Unable to switch stdout to binary mode." );
#endif
      e57::Reader reader( path, {} );
      const auto scanCount64 = reader.GetData3DCount();
      if ( scanCount64 <= 0 || scanCount64 > std::numeric_limits<std::uint32_t>::max() )
         throw std::runtime_error( "The E57 contains no usable Data3D scans." );

      std::vector<e57::Data3D> scans( static_cast<std::size_t>( scanCount64 ) );
      std::uint64_t totalPoints = 0;
      bool allIntensity = true;
      for ( std::int64_t index = 0; index < scanCount64; ++index )
      {
         auto &header = scans[static_cast<std::size_t>( index )];
         if ( !reader.ReadData3D( index, header ) )
            throw std::runtime_error( "Failed to read an E57 Data3D header." );
         if ( !hasCartesian( header ) && !hasSpherical( header ) )
            throw std::runtime_error( "An E57 scan has neither complete Cartesian nor spherical coordinates." );
         if ( !header.pointFields.intensityField && !hasColour( header ) )
            throw std::runtime_error( "An E57 scan has neither intensity nor complete RGB fields." );
         allIntensity = allIntensity && header.pointFields.intensityField;
         totalPoints += static_cast<std::uint64_t>( header.pointCount );
      }

      Header outputHeader{};
      std::memcpy( outputHeader.magic, kMagic, sizeof( kMagic ) );
      outputHeader.scanCount = static_cast<std::uint32_t>( scanCount64 );
      outputHeader.flags = allIntensity ? 1U : 0U;
      outputHeader.pointCount = totalPoints;
      writeValue( outputHeader );
      std::cout.flush();

      std::vector<double> a( capacity ), b( capacity ), c( capacity );
      std::vector<double> intensity( capacity );
      std::vector<std::uint16_t> red( capacity ), green( capacity ), blue( capacity );
      std::vector<std::int8_t> coordinateInvalid( capacity ), intensityInvalid( capacity ), colourInvalid( capacity );
      std::vector<double> outX( capacity ), outY( capacity ), outZ( capacity ), outIntensity( capacity );

      for ( std::int64_t scanIndex = 0; scanIndex < scanCount64; ++scanIndex )
      {
         const auto &header = scans[static_cast<std::size_t>( scanIndex )];
         const bool cartesian = hasCartesian( header );
         const bool useIntensity = header.pointFields.intensityField;
         e57::Data3DPointsDouble buffers;
         if ( cartesian )
         {
            buffers.cartesianX = a.data();
            buffers.cartesianY = b.data();
            buffers.cartesianZ = c.data();
            if ( header.pointFields.cartesianInvalidStateField )
               buffers.cartesianInvalidState = coordinateInvalid.data();
         }
         else
         {
            buffers.sphericalRange = a.data();
            buffers.sphericalAzimuth = b.data();
            buffers.sphericalElevation = c.data();
            if ( header.pointFields.sphericalInvalidStateField )
               buffers.sphericalInvalidState = coordinateInvalid.data();
         }
         if ( useIntensity )
         {
            buffers.intensity = intensity.data();
            if ( header.pointFields.isIntensityInvalidField )
               buffers.isIntensityInvalid = intensityInvalid.data();
         }
         else
         {
            buffers.colorRed = red.data();
            buffers.colorGreen = green.data();
            buffers.colorBlue = blue.data();
            if ( header.pointFields.isColorInvalidField )
               buffers.isColorInvalid = colourInvalid.data();
         }

         auto pointReader = reader.SetUpData3DPointsData( scanIndex, capacity, buffers );
         while ( const auto count = pointReader.read() )
         {
            std::size_t validCount = 0;
            for ( std::size_t i = 0; i < count; ++i )
            {
               const bool badCoordinate =
                  ( cartesian && header.pointFields.cartesianInvalidStateField && coordinateInvalid[i] != 0 ) ||
                  ( !cartesian && header.pointFields.sphericalInvalidStateField && coordinateInvalid[i] != 0 );
               const bool badValue =
                  ( useIntensity && header.pointFields.isIntensityInvalidField && intensityInvalid[i] != 0 ) ||
                  ( !useIntensity && header.pointFields.isColorInvalidField && colourInvalid[i] != 0 );
               if ( badCoordinate || badValue )
                  continue;
               double x, y, z;
               if ( cartesian )
               {
                  x = a[i];
                  y = b[i];
                  z = c[i];
               }
               else
               {
                  const double horizontal = a[i] * std::cos( c[i] );
                  x = horizontal * std::cos( b[i] );
                  y = horizontal * std::sin( b[i] );
                  z = a[i] * std::sin( c[i] );
               }
               double value = useIntensity
                                 ? intensity[i]
                                 : 0.2126 * static_cast<double>( red[i] ) +
                                      0.7152 * static_cast<double>( green[i] ) +
                                      0.0722 * static_cast<double>( blue[i] );
               if ( !std::isfinite( x ) || !std::isfinite( y ) || !std::isfinite( z ) ||
                    !std::isfinite( value ) )
                  continue;
               applyPose( header.pose, x, y, z );
               outX[validCount] = x;
               outY[validCount] = y;
               outZ[validCount] = z;
               outIntensity[validCount] = value;
               ++validCount;
            }
            if ( validCount > 0 )
            {
               const auto frameCount = static_cast<std::uint64_t>( validCount );
               writeValue( frameCount );
               writeArray( outX, validCount );
               writeArray( outY, validCount );
               writeArray( outZ, validCount );
               writeArray( outIntensity, validCount );
               std::cout.flush();
            }
         }
         pointReader.close();
      }
      const std::uint64_t end = 0;
      writeValue( end );
      std::cout.flush();
      return 0;
   }
}

#ifdef _WIN32
int wmain( int argc, wchar_t **argv )
{
   try
   {
      if ( argc != 3 )
         throw std::runtime_error( "Usage: OrthoE57Reader.exe <input.e57> <buffer-capacity>" );
      return run( utf8( argv[1] ), static_cast<std::size_t>( std::stoull( argv[2] ) ) );
   }
   catch ( const std::exception &error )
   {
      std::cerr << error.what() << std::endl;
      return 2;
   }
}
#else
int main( int argc, char **argv )
{
   try
   {
      if ( argc != 3 )
         throw std::runtime_error( "Usage: OrthoE57Reader <input.e57> <buffer-capacity>" );
      return run( argv[1], static_cast<std::size_t>( std::stoull( argv[2] ) ) );
   }
   catch ( const std::exception &error )
   {
      std::cerr << error.what() << std::endl;
      return 2;
   }
}
#endif
