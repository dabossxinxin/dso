/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

 
#include "util/NumType.h"

namespace dso
{
	struct RawResidualJacobian
	{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		VecNRf resF;		// 关键点的光度残差[8x1]
		Vec6f Jpdxi[2];		// 关键点坐标对帧位姿的雅可比[6x2]
		VecCf Jpdc[2];		// 关键点坐标对帧相机内参的雅可比[4x2]
		Vec2f Jpdd;			// 关键点坐标对关键帧逆深度的雅可比[2x1]
		VecNRf JIdx[2];		// 关键点光度误差对关键点坐标的雅可比[8x2]
		VecNRf JabF[2];		// 关键点光度误差对光度参数ab的雅可比[8x2]

		Mat22f JIdx2;		// JIdx^T * JIdx[2x2]
		Mat22f JabJIdx;		// Jab^T * JIdx[2x2]
		Mat22f Jab2;		// Jab^T * Jab[2x2]
	};
}